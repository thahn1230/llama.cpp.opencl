#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

typedef uchar uint8_t;

#define QK4_0 32

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_O_DATA4(x) (x)

#ifndef DK
#define DK 128
#endif

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)

// Q4_0 dequantization - EXACT pattern from cpy.cl
// Q4_0 format: qs[i] stores value[i] (low nibble) and value[i+16] (high nibble)
// Cooperative loading for K cache
inline void dequant_q4_0_block_cooperative(
    global const struct block_q4_0 * block,
    __local half4 * out,
    int tid,
    int stride
) {
    const half d = block->d;
    
    // 32 elements = 16 qs bytes = 8 half4 vectors
    // Process cooperatively across threads
    for (int vec_idx = tid; vec_idx < 8; vec_idx += stride) {
        // Which 4 elements are we processing?
        const int elem_base = vec_idx * 4;
        
        if (elem_base < 16) {
            // First 4 half4 vectors: elements 0-15 (low nibbles)
            const int qs_base = elem_base;
            out[vec_idx] = (half4)(
                (half)(((int)(block->qs[qs_base + 0] & 0x0F) - 8) * d),
                (half)(((int)(block->qs[qs_base + 1] & 0x0F) - 8) * d),
                (half)(((int)(block->qs[qs_base + 2] & 0x0F) - 8) * d),
                (half)(((int)(block->qs[qs_base + 3] & 0x0F) - 8) * d)
            );
        } else {
            // Last 4 half4 vectors: elements 16-31 (high nibbles)
            const int qs_base = elem_base - 16;
            out[vec_idx] = (half4)(
                (half)(((int)(block->qs[qs_base + 0] >> 4) - 8) * d),
                (half)(((int)(block->qs[qs_base + 1] >> 4) - 8) * d),
                (half)(((int)(block->qs[qs_base + 2] >> 4) - 8) * d),
                (half)(((int)(block->qs[qs_base + 3] >> 4) - 8) * d)
            );
        }
    }
}

// Fast inline V dequantization - EXACT pattern from cpy.cl
inline half4 dequant_q4_0_vec4(
    global const struct block_q4_0 * blocks,
    int elem_idx
) {
    const int block_idx = elem_idx / QK4_0;
    const int offset_in_block = elem_idx % QK4_0;
    
    global const struct block_q4_0 * block = &blocks[block_idx];
    const half d = block->d;
    
    // Each half4 has 4 consecutive elements
    if (offset_in_block < 16) {
        // Elements 0-15: low nibbles
        return (half4)(
            (half)(((int)(block->qs[offset_in_block + 0] & 0x0F) - 8) * d),
            (half)(((int)(block->qs[offset_in_block + 1] & 0x0F) - 8) * d),
            (half)(((int)(block->qs[offset_in_block + 2] & 0x0F) - 8) * d),
            (half)(((int)(block->qs[offset_in_block + 3] & 0x0F) - 8) * d)
        );
    } else {
        // Elements 16-31: high nibbles
        const int qs_idx = offset_in_block - 16;
        return (half4)(
            (half)(((int)(block->qs[qs_idx + 0] >> 4) - 8) * d),
            (half)(((int)(block->qs[qs_idx + 1] >> 4) - 8) * d),
            (half)(((int)(block->qs[qs_idx + 2] >> 4) - 8) * d),
            (half)(((int)(block->qs[qs_idx + 3] >> 4) - 8) * d)
        );
    }
}

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return pow(base, exph);
}

// =================================================================================================
// Optimized Flash Attention for Q4_0 KV Cache
// =================================================================================================
// Strategy:
// 1. K only in local memory (avoid local memory overflow)
// 2. V on-the-fly with vectorized dequantization
// 3. Minimize dequantization overhead with SIMD operations
// =================================================================================================
__kernel void flash_attn_f32_q4_0(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid            = get_local_id(0);
    const int block_q_idx    = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + tid;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx  = head_batch_idx % n_head;

    const int gqa_ratio   = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global       char* o_base = (global       char*)o_void + o_offset;

    const bool valid_query = (my_query_row < n_q);

    // Mask pointer
    const global MASK_DATA_TYPE* mask_ptr = NULL;
    if (mask_void != NULL && valid_query) {
        const int mask_head_idx  = head_idx  % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        const global char* mask_base =
            (const global char*)mask_void + mask_offset
            + (ulong)mask_batch_idx * mask_nb3
            + (ulong)mask_head_idx  * mask_nb2
            + (ulong)my_query_row   * mask_nb1;
        mask_ptr = (const global MASK_DATA_TYPE*)mask_base;
    }

    // Cache Q in float precision
    float4 q_priv[DK_VEC];
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = (float4)(0.0f);
    }
    
    if (valid_query) {
        const ulong q_row_offset =
            (ulong)batch_idx    * q_nb3 +
            (ulong)head_idx     * q_nb2 +
            (ulong)my_query_row * q_nb1;
        const global float4* q_ptr = (const global float4*)(q_base + q_row_offset);

        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) {
            q_priv[i] = q_ptr[i];
        }
    }

    // Output accumulator
    float4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (float4)(0.0f);
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);
    const int causal_limit = is_causal ? (n_kv - n_q + my_query_row) : n_kv;

    // Only K in local memory (8KB) - better occupancy for Q4_0
    // V is dequantized on-the-fly with optimized inline function
    __local half4 l_k[BLOCK_N][DK_VEC];

    // Precompute base offsets
    const ulong v_base_offset = (ulong)batch_idx * v_nb3 + (ulong)head_kv_idx * v_nb2;
    const ulong k_base_offset = (ulong)batch_idx * k_nb3 + (ulong)head_kv_idx * k_nb2;

    // K/V are Q4_0: each row has DK/32 blocks
    const int k_blocks_per_row = DK / QK4_0;
    const int v_blocks_per_row = DV / QK4_0;

    // Main KV loop
    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        const int k_tile_end = min(k_start + BLOCK_N, n_kv);
        const int k_tile_size = k_tile_end - k_start;
        
        // Load and dequantize K tile cooperatively
        #pragma unroll 1
        for (int row = 0; row < BLOCK_N; ++row) {
            if (row < k_tile_size) {
                const ulong k_row_offset = k_base_offset + (k_start + row) * k_nb1;
                const global struct block_q4_0 * k_blocks = 
                    (const global struct block_q4_0 *)(k_base + k_row_offset);
                
                for (int block_idx = 0; block_idx < k_blocks_per_row; ++block_idx) {
                    dequant_q4_0_block_cooperative(
                        &k_blocks[block_idx],
                        &l_k[row][block_idx * (QK4_0/4)],
                        tid,
                        WG_SIZE
                    );
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (valid_query) {
            // Process KV tokens
            #if BLOCK_N >= 32
                #define UNROLL_FACTOR 4
            #elif BLOCK_N >= 16  
                #define UNROLL_FACTOR 4
            #else
                #define UNROLL_FACTOR 2
            #endif
            
            #pragma unroll 1
            for (int j = 0; j < k_tile_size; j += UNROLL_FACTOR) {
                // Compute attention scores
                float scores[UNROLL_FACTOR];
                
                #pragma unroll
                for (int w = 0; w < UNROLL_FACTOR; ++w) {
                    if (j + w < k_tile_size) {
                        float score = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < DK_VEC; ++k) {
                            score += dot(q_priv[k], convert_float4(l_k[j + w][k]));
                        }
                        scores[w] = score * scale;
                    } else {
                        scores[w] = -INFINITY;
                    }
                }

                // Apply masking and bias
                #pragma unroll
                for (int w = 0; w < UNROLL_FACTOR; ++w) {
                    const int k_row = k_start + j + w;
                    
                    if (k_row >= n_kv || k_row > causal_limit) {
                        scores[w] = -INFINITY;
                    } else {
                        if (mask_ptr != NULL) {
                            scores[w] += slope * (float)mask_ptr[k_row];
                        }
                        if (logit_softcap > 0.0f) {
                            scores[w] = logit_softcap * tanh(scores[w] / logit_softcap);
                        }
                    }
                }

                // Online softmax update
                float m_new = m_i;
                #pragma unroll
                for (int w = 0; w < UNROLL_FACTOR; ++w) {
                    m_new = fmax(m_new, scores[w]);
                }

                const float scale_prev = exp(m_i - m_new);
                
                float p[UNROLL_FACTOR];
                float p_sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < UNROLL_FACTOR; ++w) {
                    p[w] = exp(scores[w] - m_new);
                    p_sum += p[w];
                }

                // V accumulation - on-the-fly dequantization with optimized inline function
                #pragma unroll
                for (int i = 0; i < DV_VEC; ++i) {
                    float4 v_acc = (float4)(0.0f);
                    
                    #pragma unroll
                    for (int w = 0; w < UNROLL_FACTOR; ++w) {
                        const int k_row = k_start + j + w;
                        if (k_row < n_kv && p[w] > 0.0f) {
                            const ulong v_row_offset = v_base_offset + k_row * v_nb1;
                            const global struct block_q4_0 * v_blocks = 
                                (const global struct block_q4_0 *)(v_base + v_row_offset);
                            
                            // Fast inline dequantization
                            const half4 v_val = dequant_q4_0_vec4(v_blocks, i * 4);
                            v_acc = mad((float4)(p[w]), convert_float4(v_val), v_acc);
                        }
                    }
                    
                    o_acc[i] = mad(v_acc, (float4)(1.0f), o_acc[i] * scale_prev);
                }

                // Update normalizer
                l_i = l_i * scale_prev + p_sum;
                m_i = m_new;
            }
            
            #undef UNROLL_FACTOR
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Output write
    if (valid_query && l_i > 0.0f) {
        // Sink handling
        if (sinks_void != NULL) {
            const global float* sinks_ptr =
                (const global float*)((const global char*)sinks_void + sinks_offset);
            const float m_sink  = sinks_ptr[head_idx];
            const float m_final = fmax(m_i, m_sink);

            const float scale_o = exp(m_i - m_final);
            const float sink_contrib = exp(m_sink - m_final);

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] *= scale_o;
            }

            l_i = l_i * scale_o + sink_contrib;
        }

        // Normalize & write
        const ulong o_row_offset =
            (ulong)batch_idx    * o_nb3 +
            (ulong)my_query_row * o_nb2 +
            (ulong)head_idx     * o_nb1;
        global float4* o_row = (global float4*)(o_base + o_row_offset);

        const float l_inv = 1.0f / l_i;
        
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) {
            o_row[i] = o_acc[i] * l_inv;
        }
    } else if (valid_query) {
        // Handle zero attention case
        const ulong o_row_offset =
            (ulong)batch_idx    * o_nb3 +
            (ulong)my_query_row * o_nb2 +
            (ulong)head_idx     * o_nb1;
        global float4* o_row = (global float4*)(o_base + o_row_offset);
        
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) {
            o_row[i] = (float4)(0.0f);
        }
    }
}
