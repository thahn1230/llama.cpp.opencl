#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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

//------------------------------------------------------------------------------
// dequantize_block_q4_0 - Optimized Q4_0 dequantization (CUDA-inspired)
//------------------------------------------------------------------------------
// Key optimization: (q - 8) * d = q * d + (-8 * d)
// Uses MAD instruction for better performance
//------------------------------------------------------------------------------
void dequantize_block_q4_0(global const struct block_q4_0 * block, float * result) {
    const float d = block->d;
    const float dm = -8.0f * d;  // Pre-compute -8*d
    
    for (int i = 0; i < QK4_0/2; ++i) {
        const uint8_t q_byte = block->qs[i];
        
        // Low nibble (first half)
        result[i] = mad((float)(q_byte & 0x0F), d, dm);
        
        // High nibble (second half)
        result[i + QK4_0/2] = mad((float)(q_byte >> 4), d, dm);
    }
}

//------------------------------------------------------------------------------
// cpy
//------------------------------------------------------------------------------

kernel void kernel_cpy_f16_f16(
        global half * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global half * dst_data = (global half *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global const half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f16_f32(
        global half * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {

    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f16(
        global float * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global half * dst_data = (global half *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_q4_0_q4_0(
        global char * src0,
        ulong offset0,
        global char * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    dst = dst + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global struct block_q4_0 * dst_data = (global struct block_q4_0 *) (dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    // Copy Q4_0 blocks
    int nblocks = ne00 / QK4_0;
    for (int i00 = get_local_id(0); i00 < nblocks; i00 += get_local_size(0)) {
        global const struct block_q4_0 * src_block = (global const struct block_q4_0 *)(src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        
        dst_data[i00].d = src_block->d;
        for (int j = 0; j < QK4_0/2; ++j) {
            dst_data[i00].qs[j] = src_block->qs[j];
        }
    }
}

kernel void kernel_cpy_q4_0_f16(
        global char * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    dst = (global half*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global half * dst_data = (global half *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    // Dequantize Q4_0 blocks to F16 (optimized)
    int nblocks = ne00 / QK4_0;
    for (int iblock = get_local_id(0); iblock < nblocks; iblock += get_local_size(0)) {
        global const struct block_q4_0 * src_block = (global const struct block_q4_0 *)(src0 + i03*nb03 + i02*nb02 + i01*nb01 + iblock*nb00);
        
        const float d = src_block->d;
        const float dm = -8.0f * d;
        
        for (int i = 0; i < QK4_0/2; ++i) {
            const uint8_t q_byte = src_block->qs[i];
            
            dst_data[iblock*QK4_0 + i] = (half)mad((float)(q_byte & 0x0F), d, dm);
            dst_data[iblock*QK4_0 + i + QK4_0/2] = (half)mad((float)(q_byte >> 4), d, dm);
        }
    }
}

kernel void kernel_cpy_q4_0_f32(
        global char * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    // Dequantize Q4_0 blocks to F32 (optimized)
    int nblocks = ne00 / QK4_0;
    for (int iblock = get_local_id(0); iblock < nblocks; iblock += get_local_size(0)) {
        global const struct block_q4_0 * src_block = (global const struct block_q4_0 *)(src0 + i03*nb03 + i02*nb02 + i01*nb01 + iblock*nb00);
        
        const float d = src_block->d;
        const float dm = -8.0f * d;
        
        for (int i = 0; i < QK4_0/2; ++i) {
            const uint8_t q_byte = src_block->qs[i];
            
            dst_data[iblock*QK4_0 + i] = mad((float)(q_byte & 0x0F), d, dm);
            dst_data[iblock*QK4_0 + i + QK4_0/2] = mad((float)(q_byte >> 4), d, dm);
        }
    }
}
