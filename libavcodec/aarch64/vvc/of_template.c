/*
 * Copyright (c) 2024 Zhao Zhili <quinkblack@foxmail.com>
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

#include "libavcodec/bit_depth_template.c"

void ff_vvc_prof_grad_filter_8x_neon(int16_t *gradient_h,
                                     int16_t *gradient_v,
                                     const ptrdiff_t gradient_stride,
                                     const int16_t *_src,
                                     const ptrdiff_t src_stride,
                                     const int width, const int height);

void ff_vvc_derive_bdof_vx_vy_neon(
        const int16_t *_src0, const int16_t *_src1, int pad_mask,
        const int16_t **gradient_h, const int16_t **gradient_v,
        int *vx, int *vy);

void FUNC2(ff_vvc_apply_bdof_min_block, BIT_DEPTH, _neon)(pixel* dst,
        const ptrdiff_t dst_stride, const int16_t *src0, const int16_t *src1,
        const int16_t **gh, const int16_t **gv, const int vx, const int vy);

static void FUNC(apply_bdof)(uint8_t *_dst, const ptrdiff_t _dst_stride,
        const int16_t *_src0, const int16_t *_src1,
        const int block_w, const int block_h)
{
    int16_t gradient_h[2][BDOF_BLOCK_SIZE * BDOF_BLOCK_SIZE];
    int16_t gradient_v[2][BDOF_BLOCK_SIZE * BDOF_BLOCK_SIZE];
    int vx, vy;
    const ptrdiff_t dst_stride  = _dst_stride / sizeof(pixel);
    pixel* dst                  = (pixel*)_dst;

    ff_vvc_prof_grad_filter_8x_neon(gradient_h[0], gradient_v[0], BDOF_BLOCK_SIZE,
                           _src0, MAX_PB_SIZE, block_w, block_h);
    ff_vvc_prof_grad_filter_8x_neon(gradient_h[1], gradient_v[1], BDOF_BLOCK_SIZE,
                           _src1, MAX_PB_SIZE, block_w, block_h);

    for (int y = 0; y < block_h; y += BDOF_MIN_BLOCK_SIZE) {
        for (int x = 0; x < block_w; x += BDOF_MIN_BLOCK_SIZE) {
            const int16_t* src0 = _src0 + y * MAX_PB_SIZE + x;
            const int16_t* src1 = _src1 + y * MAX_PB_SIZE + x;
            pixel *d            = dst + x;
            const int idx       = BDOF_BLOCK_SIZE * y  + x;
            const int16_t* gh[] = { gradient_h[0] + idx, gradient_h[1] + idx };
            const int16_t* gv[] = { gradient_v[0] + idx, gradient_v[1] + idx };
            const int pad_mask = !x | ((!y) << 1) |
                        ((x + BDOF_MIN_BLOCK_SIZE == block_w) << 2) |
                        ((y + BDOF_MIN_BLOCK_SIZE == block_h) << 3);
            ff_vvc_derive_bdof_vx_vy_neon(src0, src1, pad_mask, gh, gv, &vx, &vy);
            FUNC2(ff_vvc_apply_bdof_min_block, BIT_DEPTH, _neon)(d, dst_stride, src0, src1, gh, gv, vx, vy);
        }
        dst += BDOF_MIN_BLOCK_SIZE * dst_stride;
    }
}
