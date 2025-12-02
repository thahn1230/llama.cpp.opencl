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

// Decoding Kernel을 위한 전용 Wave Size (Adreno 최적화: 64)
#define DEC_WG_SIZE 64

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
// Optimized Prefill Kernel - FlashAttention Standard Implementation
// =================================================================================================
// 최적화 기법:
// 1. K와 V 모두 local memory 캐싱 - FlashAttention 원칙 준수
// 2. Q를 float으로 유지 - 불필요한 변환 제거
// 3. Adaptive unrolling - BLOCK_N 크기에 따라 최적화
// 4. Simplified indexing - division/modulo 연산 최소화
// 5. Better memory coalescing - 연속 메모리 접근 패턴
// 6. Reduced barriers - 동기화 오버헤드 최소화
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

    // Mask pointer 사전 계산
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

    // Q를 float precision으로 레지스터에 캐시
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

    // K와 V 모두 local memory에 캐싱 (FlashAttention 표준)
    __local half4 l_k[BLOCK_N][DK_VEC];
    __local half4 l_v[BLOCK_N][DV_VEC];

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
        
        // K 타일을 협력적으로 로드 (coalesced access)
        #pragma unroll 1
        for (int idx = tid; idx < BLOCK_N * DK_VEC; idx += WG_SIZE) {
            const int row = idx / DK_VEC;
            const int col = idx % DK_VEC;
            
            half4 k_val = (half4)(0.0h);
            if (row < k_tile_size) {
                const ulong k_row_offset = k_base_offset + (k_start + row) * k_nb1;
                const global struct block_q4_0 * k_blocks = 
                    (const global struct block_q4_0 *)(k_base + k_row_offset);
                
                const int block_idx = col / 8;
                const int vec_idx = col % 8;
                const global struct block_q4_0 * kb = &k_blocks[block_idx];
                const half d = kb->d;
                
                const int elem_base = vec_idx * 4;
                if (vec_idx < 4) {
                    k_val = (half4)(
                        (half)(((int)(kb->qs[elem_base + 0] & 0x0F) - 8) * d),
                        (half)(((int)(kb->qs[elem_base + 1] & 0x0F) - 8) * d),
                        (half)(((int)(kb->qs[elem_base + 2] & 0x0F) - 8) * d),
                        (half)(((int)(kb->qs[elem_base + 3] & 0x0F) - 8) * d)
                    );
                } else {
                    const int qs_base = elem_base - 16;
                    k_val = (half4)(
                        (half)(((int)(kb->qs[qs_base + 0] >> 4) - 8) * d),
                        (half)(((int)(kb->qs[qs_base + 1] >> 4) - 8) * d),
                        (half)(((int)(kb->qs[qs_base + 2] >> 4) - 8) * d),
                        (half)(((int)(kb->qs[qs_base + 3] >> 4) - 8) * d)
                    );
                }
            }
            l_k[row][col] = k_val;
        }

        // V 타일을 협력적으로 로드하고 dequantize (coalesced access)
        #pragma unroll 1
        for (int idx = tid; idx < BLOCK_N * DV_VEC; idx += WG_SIZE) {
            const int row = idx / DV_VEC;
            const int col = idx % DV_VEC;
            
            half4 v_val = (half4)(0.0h);
            if (row < k_tile_size) {
                const ulong v_row_offset = v_base_offset + (k_start + row) * v_nb1;
                const global struct block_q4_0 * v_blocks = 
                    (const global struct block_q4_0 *)(v_base + v_row_offset);
                
                const int block_idx = col / 8;
                const int vec_idx = col % 8;
                const global struct block_q4_0 * vb = &v_blocks[block_idx];
                const half d = vb->d;
                
                const int elem_base = vec_idx * 4;
                if (vec_idx < 4) {
                    v_val = (half4)(
                        (half)(((int)(vb->qs[elem_base + 0] & 0x0F) - 8) * d),
                        (half)(((int)(vb->qs[elem_base + 1] & 0x0F) - 8) * d),
                        (half)(((int)(vb->qs[elem_base + 2] & 0x0F) - 8) * d),
                        (half)(((int)(vb->qs[elem_base + 3] & 0x0F) - 8) * d)
                    );
                } else {
                    const int qs_base = elem_base - 16;
                    v_val = (half4)(
                        (half)(((int)(vb->qs[qs_base + 0] >> 4) - 8) * d),
                        (half)(((int)(vb->qs[qs_base + 1] >> 4) - 8) * d),
                        (half)(((int)(vb->qs[qs_base + 2] >> 4) - 8) * d),
                        (half)(((int)(vb->qs[qs_base + 3] >> 4) - 8) * d)
                    );
                }
            }
            l_v[row][col] = v_val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (valid_query) {
            // Process KV tokens - adaptive unrolling based on BLOCK_N
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

                // V accumulation - local memory access (FlashAttention)
                #pragma unroll
                for (int i = 0; i < DV_VEC; ++i) {
                    float4 v_acc = (float4)(0.0f);
                    
                    #pragma unroll
                    for (int w = 0; w < UNROLL_FACTOR; ++w) {
                        if (j + w < k_tile_size && p[w] > 0.0f) {
                            v_acc = mad((float4)(p[w]), convert_float4(l_v[j + w][i]), v_acc);
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
        // Sink 처리
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


// =================================================================================================
// 2. Optimized Decoding Kernel (Dimension Parallelism for Adreno)
// =================================================================================================
__kernel void flash_attn_f32_q4_0_q1(
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
    // Q를 Shared Memory에 캐싱 (Half precision으로 변환하여 저장)
    // DK는 128 같은 매크로 상수로 정의되어야 함.
    __local half l_q[DK];

    const int tid = get_local_id(0); // 0 ~ 63
    const int head_batch_idx = get_global_id(1);
    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;
    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    // [1] Load Q: 64개 스레드가 협력하여 DK(128) 크기의 Q를 로드
    const global char* q_base = (const global char*)q_void + q_offset;
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global float* q_ptr = (const global float*)(q_base + q_row_offset);

    // DEC_WG_SIZE(64) 스트라이드로 로드
    for (int i = tid; i < DK; i += DEC_WG_SIZE) {
        l_q[i] = (half)q_ptr[i]; 
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // [2] Accumulator 초기화
    // 레지스터 스필 방지를 위해 각 스레드는 오직 2개의 Output(float2)만 관리
    // DK=128, WG=64 -> 1 thread covers 2 elements (index: tid*2, tid*2+1)
    float2 my_o_acc = (float2)(0.0f, 0.0f);
    
    float m_local = -INFINITY;
    float l_local = 0.0f;

    const int my_dim_base = tid * 2; // 내 스레드가 담당할 차원 시작점

    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    
    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);
    
    // Sink 초기화
    if (sinks_void != NULL) {
        const global float* sinks_ptr = (const global float*)((const global char*)sinks_void + sinks_offset);
        m_local = sinks_ptr[head_idx];
    }

    // [3] Main Loop (Iterate over KV tokens)
    // 스레드당 연산량을 최소화하여 GPU 점유율 극대화
    for (int k_idx = 0; k_idx < n_kv; ++k_idx) {
        // A. Partial Dot Product (내 담당 차원만 계산)
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        
        // K를 dequantize하여 읽기 (2개 원소)
        const global struct block_q4_0 * k_blocks = 
            (const global struct block_q4_0 *)(k_base + k_row_offset);
        
        // 범위 체크 (DK가 128이 아닐 경우 대비)
        float my_score_part = 0.0f;
        if (my_dim_base < DK) {
            // Local Memory Q 읽기 (half* -> half2 -> float2)
            half2 q_h2 = *(__local half2*)&l_q[my_dim_base];
            float2 q_val = convert_float2(q_h2);
            
            // Dequantize 2 K elements
            const int block_idx = my_dim_base / QK4_0;
            const int offset_in_block = my_dim_base % QK4_0;
            const global struct block_q4_0 * kb = &k_blocks[block_idx];
            const float d = kb->d;
            
            float2 k_val;
            if (offset_in_block < 16) {
                k_val = (float2)(
                    ((int)(kb->qs[offset_in_block] & 0x0F) - 8) * d,
                    ((int)(kb->qs[offset_in_block + 1] & 0x0F) - 8) * d
                );
            } else {
                const int qs_idx = offset_in_block - 16;
                k_val = (float2)(
                    ((int)(kb->qs[qs_idx] >> 4) - 8) * d,
                    ((int)(kb->qs[qs_idx + 1] >> 4) - 8) * d
                );
            }
            
            my_score_part = dot(q_val, k_val);
        }

        // B. Reduction (Subgroup Sum) -> 전체 차원(128)에 대한 Score 완성
        // Adreno 하드웨어 레벨 리덕션 (매우 빠름)
        float score = sub_group_reduce_add(my_score_part);
        
        // Score Scaling & Masking (모든 스레드가 동일한 값 보유)
        score *= scale;
        if (mask_base != NULL) {
            const global half* mask_ptr = (const global half*)(mask_base);
            score += slope * (float)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }

        // C. Online Softmax Update
        float m_prev = m_local;
        m_local = max(m_prev, score);
        
        float p = 0.0f;
        float scale_prev = 1.0f;
        if (m_local > -INFINITY) {
            p = exp(score - m_local);
            scale_prev = (m_prev > -INFINITY) ? exp(m_prev - m_local) : 0.0f;
        }

        l_local = l_local * scale_prev + p;

        // D. Accumulate V (Dimension Parallel)
        // 내 담당 차원(2개)에 해당하는 V값만 업데이트
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global struct block_q4_0 * v_blocks = 
            (const global struct block_q4_0 *)(v_base + v_row_offset);
        
        if (my_dim_base < DV) {
            const int block_idx = my_dim_base / QK4_0;
            const int offset_in_block = my_dim_base % QK4_0;
            const global struct block_q4_0 * vb = &v_blocks[block_idx];
            const float d = vb->d;
            
            float2 v_val;
            if (offset_in_block < 16) {
                v_val = (float2)(
                    ((int)(vb->qs[offset_in_block] & 0x0F) - 8) * d,
                    ((int)(vb->qs[offset_in_block + 1] & 0x0F) - 8) * d
                );
            } else {
                const int qs_idx = offset_in_block - 16;
                v_val = (float2)(
                    ((int)(vb->qs[qs_idx] >> 4) - 8) * d,
                    ((int)(vb->qs[qs_idx + 1] >> 4) - 8) * d
                );
            }
            
            // my_o = my_o * scale + p * v
            my_o_acc = mad((float2)(p), v_val, my_o_acc * scale_prev);
        }
    }

    // [4] Final Normalize & Write
    if (l_local > 0.0f) {
        float l_inv = 1.0f / l_local;
        my_o_acc *= l_inv;
    } else {
        my_o_acc = (float2)(0.0f);
    }

    // Global Memory Write (각 스레드가 2개씩 씀)
    global char* o_base = (global char*)o_void + o_offset;
    ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global float* o_ptr = (global float*)(o_base + o_row_offset);

    if (my_dim_base < DV) {
        // float2로 기록하는 것이 대역폭에 유리할 수 있으나,
        // o_ptr이 float* 이므로 개별 기록 (컴파일러가 병합 최적화 수행함)
        o_ptr[my_dim_base] = my_o_acc.x;
        if (my_dim_base + 1 < DV) {
            o_ptr[my_dim_base + 1] = my_o_acc.y;
        }
    }
}