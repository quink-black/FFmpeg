/*
 * Copyright (c) 2024 Zhao Zhili
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <stdint.h>
#include <wasm_simd128.h>

#include "libavutil/cpu_internal.h"
#include "libavcodec/hevc/dsp.h"

#if HAVE_SIMD128
static const int8_t transform[] = {
    64, 83, 64, 36, 89, 75, 50, 18,
    90, 87, 80, 70, 57, 43, 25, 9,
    90, 90, 88, 85, 82, 78, 73, 67,
    61, 54, 46, 38, 31, 22, 13, 4,
};

static inline void transpose_4x8h(v128_t *src)
{
    v128_t t0 = wasm_i16x8_shuffle(src[0], src[1], 0, 8, 2, 10, 4, 12, 6, 14);
    v128_t t1 = wasm_i16x8_shuffle(src[0], src[1], 1, 9, 3, 11, 5, 13, 7, 15);
    v128_t t2 = wasm_i16x8_shuffle(src[2], src[3], 0, 8, 2, 10, 4, 12, 6, 14);
    v128_t t3 = wasm_i16x8_shuffle(src[2], src[3], 1, 9, 3, 11, 5, 13, 7, 15);

    src[0] = wasm_i32x4_shuffle(t0, t2, 0, 4, 2, 6);
    src[2] = wasm_i32x4_shuffle(t0, t2, 1, 5, 3, 7);
    src[1] = wasm_i32x4_shuffle(t1, t3, 0, 4, 2, 6);
    src[3] = wasm_i32x4_shuffle(t1, t3, 1, 5, 3, 7);
}

static inline void transpose_8x8h(v128_t *src)
{
    transpose_4x8h(src);
    transpose_4x8h(&src[4]);
}

static inline void tr_4x4(v128_t *src, v128_t *trans, v128_t add, int shift)
{
    v128_t tmp[4];
    v128_t e0 = wasm_i32x4_extmul_low_i16x8(src[0], trans[0]);
    v128_t e1 = wasm_i32x4_extmul_low_i16x8(src[0], trans[0]);
    v128_t o0 = wasm_i32x4_extmul_low_i16x8(src[1], trans[1]);
    v128_t o1 = wasm_i32x4_extmul_low_i16x8(src[1], trans[3]);

    tmp[0] = wasm_i32x4_extmul_low_i16x8(src[2], trans[0]);
    tmp[1] = wasm_i32x4_extmul_low_i16x8(src[2], trans[0]);
    tmp[2] = wasm_i32x4_extmul_low_i16x8(src[3], trans[3]);
    tmp[3] = wasm_i32x4_extmul_low_i16x8(src[3], trans[1]);
    e0 = wasm_i32x4_add(e0, tmp[0]);
    e1 = wasm_i32x4_sub(e1, tmp[1]);
    o0 = wasm_i32x4_add(o0, tmp[2]);
    o1 = wasm_i32x4_sub(o1, tmp[3]);

    tmp[0] = wasm_i32x4_add(e0, o0);
    tmp[1] = wasm_i32x4_sub(e0, o0);
    tmp[2] = wasm_i32x4_add(e1, o1);
    tmp[3] = wasm_i32x4_sub(e1, o1);

    tmp[0] = wasm_i32x4_add(tmp[0], add);
    tmp[1] = wasm_i32x4_add(tmp[1], add);
    tmp[2] = wasm_i32x4_add(tmp[2], add);
    tmp[3] = wasm_i32x4_add(tmp[3], add);
    tmp[0] = wasm_i32x4_shr(tmp[0], shift);
    tmp[1] = wasm_i32x4_shr(tmp[1], shift);
    tmp[2] = wasm_i32x4_shr(tmp[2], shift);
    tmp[3] = wasm_i32x4_shr(tmp[3], shift);

    src[0] = wasm_i16x8_narrow_i32x4(tmp[0], tmp[0]);
    src[3] = wasm_i16x8_narrow_i32x4(tmp[1], tmp[1]);
    src[1] = wasm_i16x8_narrow_i32x4(tmp[2], tmp[2]);
    src[2] = wasm_i16x8_narrow_i32x4(tmp[3], tmp[3]);
}

static void idct_4x4(int16_t *coeffs, int bit_depth)
{
    v128_t src[4];
    v128_t trans[4];
    v128_t add;
    const int shift1 = 7;
    int shift2 = 20 - bit_depth;

    src[0] = wasm_v128_load64_zero(coeffs + 0);
    src[1] = wasm_v128_load64_zero(coeffs + 4);
    src[2] = wasm_v128_load64_zero(coeffs + 8);
    src[3] = wasm_v128_load64_zero(coeffs + 12);

    trans[0] = wasm_i16x8_const_splat(transform[0]);
    trans[1] = wasm_i16x8_const_splat(transform[1]);
    trans[2] = wasm_i16x8_const_splat(transform[2]);
    trans[3] = wasm_i16x8_const_splat(transform[3]);

    add = wasm_i32x4_const_splat((1 << (shift1 - 1)));
    tr_4x4(src, trans, add, shift1);
    transpose_4x8h(src);

    add = wasm_i32x4_splat((1 << (shift2 - 1)));
    tr_4x4(src, trans, add, shift2);
    transpose_4x8h(src);

    wasm_v128_store64_lane(coeffs + 0, src[0], 0);
    wasm_v128_store64_lane(coeffs + 4, src[1], 0);
    wasm_v128_store64_lane(coeffs + 8, src[2], 0);
    wasm_v128_store64_lane(coeffs + 12, src[3], 0);
}

static void idct_4x4_8(int16_t *coeffs, int col_limit)
{
    idct_4x4(coeffs, 8);
}

static void idct_4x4_10(int16_t *coeffs, int col_limit)
{
    idct_4x4(coeffs, 10);
}

static inline void shift_narrow(v128_t src, v128_t *dst, v128_t add,
                                bool lower, int shift)
{
    v128_t zero = wasm_i32x4_const_splat(0);

    src = wasm_i32x4_add(src, add);
    src = wasm_i32x4_shr(src, shift);

    if (lower)
        *dst = wasm_i16x8_narrow_i32x4(src, zero);
    else
        *dst = wasm_i16x8_narrow_i32x4(zero, src);
}

static void tr_4x4_8(v128_t in0, v128_t in1, v128_t in2, v128_t in3,
                     v128_t *dst0, v128_t *dst1, v128_t *dst2, v128_t *dst3,
                     const v128_t *trans, bool lower0, bool lower1)
{
    v128_t e0, e1, o0, o1;
    v128_t tmp[4];

    if (lower0) {
        e0 = wasm_i32x4_extmul_low_i16x8(in0, trans[0]);
        e1 = e0;
        o0 = wasm_i32x4_extmul_low_i16x8(in1, trans[1]);
        o1 = wasm_i32x4_extmul_low_i16x8(in1, trans[3]);
    } else {
        e0 = wasm_i32x4_extmul_high_i16x8(in0, trans[0]);
        e1 = e0;
        o0 = wasm_i32x4_extmul_high_i16x8(in1, trans[1]);
        o1 = wasm_i32x4_extmul_high_i16x8(in1, trans[3]);
    }

    if (lower1) {
        tmp[0] = wasm_i32x4_extmul_low_i16x8(in2, trans[0]);
        tmp[1] = wasm_i32x4_extmul_low_i16x8(in2, trans[0]);
        tmp[2] = wasm_i32x4_extmul_low_i16x8(in3, trans[3]);
        tmp[3] = wasm_i32x4_extmul_low_i16x8(in3, trans[1]);
    } else {
        tmp[0] = wasm_i32x4_extmul_high_i16x8(in2, trans[0]);
        tmp[1] = wasm_i32x4_extmul_high_i16x8(in2, trans[0]);
        tmp[2] = wasm_i32x4_extmul_high_i16x8(in3, trans[3]);
        tmp[3] = wasm_i32x4_extmul_high_i16x8(in3, trans[1]);
    }

    e0 = wasm_i32x4_add(e0, tmp[0]);
    e1 = wasm_i32x4_sub(e1, tmp[1]);
    o0 = wasm_i32x4_add(o0, tmp[2]);
    o1 = wasm_i32x4_sub(o1, tmp[3]);

    *dst0 = wasm_i32x4_add(e0, o0);
    *dst1 = wasm_i32x4_add(e1, o1);
    *dst2 = wasm_i32x4_sub(e1, o1);
    *dst3 = wasm_i32x4_sub(e0, o0);
}

/*
 * lower0: for src[0] to src[3]
 * lower1: for src[4] to src[7]
 */
static void tr_8x4(const v128_t *src0, const v128_t *src1,
                   bool lower0, bool lower1,
                   v128_t *dst, const v128_t *trans, int shift)
{
    v128_t v24, v25, v26, v27, v28, v29, v30, v31;
    v128_t add = wasm_i32x4_splat((1 << (shift - 1)));

    tr_4x4_8(src0[0], src0[2], src1[0], src1[2], &v24, &v25, &v26, &v27,
             trans, lower0, lower1);

    if (lower0) {
        v30 = wasm_i32x4_extmul_low_i16x8(src0[1], trans[6]);
        v28 = wasm_i32x4_extmul_low_i16x8(src0[1], trans[4]);
        v29 = wasm_i32x4_extmul_low_i16x8(src0[1], trans[5]);
        v30 = wasm_i32x4_sub(v30, wasm_i32x4_extmul_low_i16x8(src0[3], trans[4]));
        v28 = wasm_i32x4_add(v28, wasm_i32x4_extmul_low_i16x8(src0[3], trans[5]));
        v29 = wasm_i32x4_sub(v29, wasm_i32x4_extmul_low_i16x8(src0[3], trans[7]));
    } else {
        v30 = wasm_i32x4_extmul_high_i16x8(src0[1], trans[6]);
        v28 = wasm_i32x4_extmul_high_i16x8(src0[1], trans[4]);
        v29 = wasm_i32x4_extmul_high_i16x8(src0[1], trans[5]);
        v30 = wasm_i32x4_sub(v30, wasm_i32x4_extmul_high_i16x8(src0[3], trans[4]));
        v28 = wasm_i32x4_add(v28, wasm_i32x4_extmul_high_i16x8(src0[3], trans[5]));
        v29 = wasm_i32x4_sub(v29, wasm_i32x4_extmul_high_i16x8(src0[3], trans[7]));
    }

    if (lower1) {
        v30 = wasm_i32x4_add(v30, wasm_i32x4_extmul_low_i16x8(src1[1], trans[7]));
        v28 = wasm_i32x4_add(v28, wasm_i32x4_extmul_low_i16x8(src1[1], trans[6]));
        v29 = wasm_i32x4_sub(v29, wasm_i32x4_extmul_low_i16x8(src1[1], trans[4]));

        v30 = wasm_i32x4_add(v30, wasm_i32x4_extmul_low_i16x8(src1[3], trans[5]));
        v28 = wasm_i32x4_add(v28, wasm_i32x4_extmul_low_i16x8(src1[3], trans[7]));
        v29 = wasm_i32x4_sub(v29, wasm_i32x4_extmul_low_i16x8(src1[3], trans[6]));
    } else {
        v30 = wasm_i32x4_add(v30, wasm_i32x4_extmul_high_i16x8(src1[1], trans[7]));
        v28 = wasm_i32x4_add(v28, wasm_i32x4_extmul_high_i16x8(src1[1], trans[6]));
        v29 = wasm_i32x4_sub(v29, wasm_i32x4_extmul_high_i16x8(src1[1], trans[4]));

        v30 = wasm_i32x4_add(v30, wasm_i32x4_extmul_high_i16x8(src1[3], trans[5]));
        v28 = wasm_i32x4_add(v28, wasm_i32x4_extmul_high_i16x8(src1[3], trans[7]));
        v29 = wasm_i32x4_sub(v29, wasm_i32x4_extmul_high_i16x8(src1[3], trans[6]));
    }

    v31 = wasm_i32x4_add(v26, v30);
    v26 = wasm_i32x4_sub(v26, v30);
    shift_narrow(v31, &dst[2], add, lower0, shift);

    if (lower0) {
        v31 = wasm_i32x4_extmul_low_i16x8(src0[1], trans[7]);
        v31 = wasm_i32x4_sub(v31, wasm_i32x4_extmul_low_i16x8(src0[3], trans[6]));
    } else {
        v31 = wasm_i32x4_extmul_high_i16x8(src0[1], trans[7]);
        v31 = wasm_i32x4_sub(v31, wasm_i32x4_extmul_high_i16x8(src0[3], trans[6]));
    }

    if (lower1) {
        v31 = wasm_i32x4_add(v31, wasm_i32x4_extmul_low_i16x8(src1[1], trans[5]));
        v31 = wasm_i32x4_sub(v31, wasm_i32x4_extmul_low_i16x8(src1[3], trans[4]));
    } else {
        v31 = wasm_i32x4_add(v31, wasm_i32x4_extmul_high_i16x8(src1[1], trans[5]));
        v31 = wasm_i32x4_sub(v31, wasm_i32x4_extmul_high_i16x8(src1[3], trans[4]));
    }
    shift_narrow(v26, &dst[5], add, lower1, shift);

    v26 = wasm_i32x4_add(v24, v28);
    v24 = wasm_i32x4_sub(v24, v28);
    v28 = wasm_i32x4_add(v25, v29);
    v25 = wasm_i32x4_sub(v25, v29);
    v30 = wasm_i32x4_add(v27, v31);
    v27 = wasm_i32x4_sub(v27, v31);

    shift_narrow(v26, &dst[0], add, lower0, shift);
    shift_narrow(v24, &dst[7], add, lower1, shift);
    shift_narrow(v28, &dst[1], add, lower0, shift);
    shift_narrow(v25, &dst[6], add, lower1, shift);
    shift_narrow(v30, &dst[3], add, lower0, shift);
    shift_narrow(v27, &dst[4], add, lower1, shift);
}

static void tr_8x8(v128_t *src, const v128_t *trans, int shift)
{
    v128_t dst0[8];

    if (shift == 7) {
        v128_t dst1[8];

        // first pass
        tr_8x4(src, &src[4], true, true, dst0, trans, shift);
        tr_8x4(src, &src[4], false, false, dst1, trans, shift);
        src[0] = wasm_i16x8_add(dst0[0], dst1[0]);
        src[1] = wasm_i16x8_add(dst0[1], dst1[1]);
        src[2] = wasm_i16x8_add(dst0[2], dst1[2]);
        src[3] = wasm_i16x8_add(dst0[3], dst1[3]);
        src[4] = wasm_i16x8_add(dst0[4], dst1[4]);
        src[5] = wasm_i16x8_add(dst0[5], dst1[5]);
        src[6] = wasm_i16x8_add(dst0[6], dst1[6]);
        src[7] = wasm_i16x8_add(dst0[7], dst1[7]);
    } else {
        tr_8x4(src, src, true, false, dst0, trans, shift);
        src[0] = wasm_i16x8_add(dst0[0], dst0[4]);
        src[1] = wasm_i16x8_add(dst0[1], dst0[5]);
        src[2] = wasm_i16x8_add(dst0[2], dst0[6]);
        src[3] = wasm_i16x8_add(dst0[3], dst0[7]);

        src += 4;
        tr_8x4(src, src, true, false, dst0, trans, shift);
        src[0] = wasm_i16x8_add(dst0[0], dst0[4]);
        src[1] = wasm_i16x8_add(dst0[1], dst0[5]);
        src[2] = wasm_i16x8_add(dst0[2], dst0[6]);
        src[3] = wasm_i16x8_add(dst0[3], dst0[7]);
    }
}

static void idct_8x8(int16_t *coeffs, int bit_depth)
{
    v128_t src[8];
    v128_t trans[8];
    const int shift1 = 7;
    int shift2 = 20 - bit_depth;

    src[0] = wasm_v128_load(coeffs + 0 * 8);
    src[1] = wasm_v128_load(coeffs + 1 * 8);
    src[2] = wasm_v128_load(coeffs + 2 * 8);
    src[3] = wasm_v128_load(coeffs + 3 * 8);
    src[4] = wasm_v128_load(coeffs + 4 * 8);
    src[5] = wasm_v128_load(coeffs + 5 * 8);
    src[6] = wasm_v128_load(coeffs + 6 * 8);
    src[7] = wasm_v128_load(coeffs + 7 * 8);

    trans[0] = wasm_i16x8_const_splat(transform[0]);
    trans[1] = wasm_i16x8_const_splat(transform[1]);
    trans[2] = wasm_i16x8_const_splat(transform[2]);
    trans[3] = wasm_i16x8_const_splat(transform[3]);
    trans[4] = wasm_i16x8_const_splat(transform[4]);
    trans[5] = wasm_i16x8_const_splat(transform[5]);
    trans[6] = wasm_i16x8_const_splat(transform[6]);
    trans[7] = wasm_i16x8_const_splat(transform[7]);

    tr_8x8(src, trans, shift1);
    transpose_8x8h(src);
    tr_8x8(src, trans, shift2);
    transpose_8x8h(src);

    wasm_v128_store(coeffs + 0 * 8, src[0]);
    wasm_v128_store(coeffs + 1 * 8, src[1]);
    wasm_v128_store(coeffs + 2 * 8, src[2]);
    wasm_v128_store(coeffs + 3 * 8, src[3]);
    wasm_v128_store(coeffs + 4 * 8, src[4]);
    wasm_v128_store(coeffs + 5 * 8, src[5]);
    wasm_v128_store(coeffs + 6 * 8, src[6]);
    wasm_v128_store(coeffs + 7 * 8, src[7]);
}

static void idct_8x8_8(int16_t *coeffs, int col_limit)
{
    idct_8x8(coeffs, 8);
}

static void idct_8x8_10(int16_t *coeffs, int col_limit)
{
    idct_8x8(coeffs, 10);
}

#endif

av_cold void ff_hevc_dsp_init_wasm(HEVCDSPContext *c, const int bit_depth)
{
    int cpu_flags = av_get_cpu_flags();
    if (!CPUEXT(cpu_flags, SIMD128))
        return;

#if HAVE_SIMD128
    if (bit_depth == 8) {
        c->idct[0] = idct_4x4_8;
        c->idct[1] = idct_8x8_8;
    } else if (bit_depth == 10) {
        c->idct[0] = idct_4x4_10;
        c->idct[1] = idct_8x8_10;
    }
#endif
}
