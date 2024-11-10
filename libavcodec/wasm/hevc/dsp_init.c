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

static inline void transpose_4x8h(v128_t *a, v128_t *b, v128_t *c, v128_t *d)
{
    v128_t t0 = wasm_i16x8_shuffle(*a, *b, 0, 8, 2, 10, 4, 12, 6, 14);
    v128_t t1 = wasm_i16x8_shuffle(*a, *b, 1, 9, 3, 11, 5, 13, 7, 15);
    v128_t t2 = wasm_i16x8_shuffle(*c, *d, 0, 8, 2, 10, 4, 12, 6, 14);
    v128_t t3 = wasm_i16x8_shuffle(*c, *d, 1, 9, 3, 11, 5, 13, 7, 15);

    *a = wasm_i32x4_shuffle(t0, t2, 0, 4, 2, 6);
    *c = wasm_i32x4_shuffle(t0, t2, 1, 5, 3, 7);
    *b = wasm_i32x4_shuffle(t1, t3, 0, 4, 2, 6);
    *d = wasm_i32x4_shuffle(t1, t3, 1, 5, 3, 7);
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
    transpose_4x8h(&src[0], &src[1], &src[2], &src[3]);

    add = wasm_i32x4_splat((1 << (shift2 - 1)));
    tr_4x4(src, trans, add, shift2);
    transpose_4x8h(&src[0], &src[1], &src[2], &src[3]);

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

#endif

av_cold void ff_hevc_dsp_init_wasm(HEVCDSPContext *c, const int bit_depth)
{
    int cpu_flags = av_get_cpu_flags();
    if (!CPUEXT(cpu_flags, SIMD128))
        return;

#if HAVE_SIMD128
    if (bit_depth == 8) {
        c->idct[0] = idct_4x4_8;
    } else if (bit_depth == 10) {
        c->idct[0] = idct_4x4_10;
    }
#endif
}
