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

#include "libavutil/aarch64/asm.S"

#define VVC_MAX_PB_SIZE 128
#define BDOF_BLOCK_SIZE 16
#define BDOF_MIN_BLOCK_SIZE 4

.macro vvc_avg type, bit_depth

.macro vvc_\type\()_\bit_depth\()_2_4 tap
.if \tap == 2
        ldr             s0, [src0]
        ldr             s2, [src1]
.else
        ldr             d0, [src0]
        ldr             d2, [src1]
.endif

.ifc \type, avg
        saddl           v4.4s, v0.4h, v2.4h
        add             v4.4s, v4.4s, v16.4s
        sqshrn          v4.4h, v4.4s, #(15 - \bit_depth)
.else
        mov             v4.16b, v16.16b
        smlal           v4.4s, v0.4h, v19.4h
        smlal           v4.4s, v2.4h, v20.4h
        sqshl           v4.4s, v4.4s, v22.4s
        sqxtn           v4.4h, v4.4s
.endif

.if \bit_depth == 8
        sqxtun          v4.8b, v4.8h
.if \tap == 2
        str             h4, [dst]
.else   // tap == 4
        str             s4, [dst]
.endif

.else   // bit_depth > 8
        smin            v4.4h, v4.4h, v17.4h
        smax            v4.4h, v4.4h, v18.4h
.if \tap == 2
        str             s4, [dst]
.else
        str             d4, [dst]
.endif
.endif
        add             src0, src0, x10
        add             src1, src1, x10
        add             dst, dst, dst_stride
.endm

function ff_vvc_\type\()_\bit_depth\()_neon, export=1
        dst             .req x0
        dst_stride      .req x1
        src0            .req x2
        src1            .req x3
        width           .req w4
        height          .req w5

        mov             x10, #(VVC_MAX_PB_SIZE * 2)
        cmp             width, #8
.ifc \type, avg
        movi            v16.4s, #(1 << (14 - \bit_depth))
.else
        lsr             x11, x6, #32        // weight0
        mov             w12, w6             // weight1
        lsr             x13, x7, #32        // offset
        mov             w14, w7             // shift

        dup             v19.8h, w11
        neg             w14, w14            // so we can use sqshl
        dup             v20.8h, w12
        dup             v16.4s, w13
        dup             v22.4s, w14
.endif // avg

 .if \bit_depth >= 10
        // clip pixel
        mov             w6, #((1 << \bit_depth) - 1)
        movi            v18.8h, #0
        dup             v17.8h, w6
.endif

        b.eq            8f
        b.hi            16f
        cmp             width, #4
        b.eq            4f
2:      // width == 2
        subs            height, height, #1
        vvc_\type\()_\bit_depth\()_2_4 2
        b.ne            2b
        b               32f
4:      // width == 4
        subs            height, height, #1
        vvc_\type\()_\bit_depth\()_2_4 4
        b.ne            4b
        b               32f
8:      // width == 8
        ld1             {v0.8h}, [src0], x10
        ld1             {v2.8h}, [src1], x10
.ifc \type, avg
        saddl           v4.4s, v0.4h, v2.4h
        saddl2          v5.4s, v0.8h, v2.8h
        add             v4.4s, v4.4s, v16.4s
        add             v5.4s, v5.4s, v16.4s
        sqshrn          v4.4h, v4.4s, #(15 - \bit_depth)
        sqshrn2         v4.8h, v5.4s, #(15 - \bit_depth)
.else
        mov             v4.16b, v16.16b
        mov             v5.16b, v16.16b
        smlal           v4.4s, v0.4h, v19.4h
        smlal           v4.4s, v2.4h, v20.4h
        smlal2          v5.4s, v0.8h, v19.8h
        smlal2          v5.4s, v2.8h, v20.8h
        sqshl           v4.4s, v4.4s, v22.4s
        sqshl           v5.4s, v5.4s, v22.4s
        sqxtn           v4.4h, v4.4s
        sqxtn2          v4.8h, v5.4s
.endif
        subs            height, height, #1
.if \bit_depth == 8
        sqxtun          v4.8b, v4.8h
        st1             {v4.8b}, [dst], dst_stride
.else
        smin            v4.8h, v4.8h, v17.8h
        smax            v4.8h, v4.8h, v18.8h
        st1             {v4.8h}, [dst], dst_stride
.endif
        b.ne            8b
        b               32f
16:     // width >= 16
        mov             w6, width
        mov             x7, src0
        mov             x8, src1
        mov             x9, dst
17:
        ldp             q0, q1, [x7], #32
        ldp             q2, q3, [x8], #32
.ifc \type, avg
        saddl           v4.4s, v0.4h, v2.4h
        saddl2          v5.4s, v0.8h, v2.8h
        saddl           v6.4s, v1.4h, v3.4h
        saddl2          v7.4s, v1.8h, v3.8h
        add             v4.4s, v4.4s, v16.4s
        add             v5.4s, v5.4s, v16.4s
        add             v6.4s, v6.4s, v16.4s
        add             v7.4s, v7.4s, v16.4s
        sqshrn          v4.4h, v4.4s, #(15 - \bit_depth)
        sqshrn2         v4.8h, v5.4s, #(15 - \bit_depth)
        sqshrn          v6.4h, v6.4s, #(15 - \bit_depth)
        sqshrn2         v6.8h, v7.4s, #(15 - \bit_depth)
.else   // avg
        mov             v4.16b, v16.16b
        mov             v5.16b, v16.16b
        mov             v6.16b, v16.16b
        mov             v7.16b, v16.16b
        smlal           v4.4s, v0.4h, v19.4h
        smlal           v4.4s, v2.4h, v20.4h
        smlal2          v5.4s, v0.8h, v19.8h
        smlal2          v5.4s, v2.8h, v20.8h
        smlal           v6.4s, v1.4h, v19.4h
        smlal           v6.4s, v3.4h, v20.4h
        smlal2          v7.4s, v1.8h, v19.8h
        smlal2          v7.4s, v3.8h, v20.8h
        sqshl           v4.4s, v4.4s, v22.4s
        sqshl           v5.4s, v5.4s, v22.4s
        sqshl           v6.4s, v6.4s, v22.4s
        sqshl           v7.4s, v7.4s, v22.4s
        sqxtn           v4.4h, v4.4s
        sqxtn           v6.4h, v6.4s
        sqxtn2          v4.8h, v5.4s
        sqxtn2          v6.8h, v7.4s
.endif  // w_avg
        subs            w6, w6, #16
.if \bit_depth == 8
        sqxtun          v4.8b, v4.8h
        sqxtun2         v4.16b, v6.8h
        str             q4, [x9], #16
.else
        smin            v4.8h, v4.8h, v17.8h
        smin            v6.8h, v6.8h, v17.8h
        smax            v4.8h, v4.8h, v18.8h
        smax            v6.8h, v6.8h, v18.8h
        stp             q4, q6, [x9], #32
.endif
        b.ne            17b

        subs            height, height, #1
        add             src0, src0, x10
        add             src1, src1, x10
        add             dst, dst, dst_stride
        b.ne            16b
32:
        ret
endfunc

.unreq dst
.unreq dst_stride
.unreq src0
.unreq src1
.unreq width
.unreq height
.endm

vvc_avg avg, 8
vvc_avg avg, 10
vvc_avg avg, 12
vvc_avg w_avg, 8
vvc_avg w_avg, 10
vvc_avg w_avg, 12

function ff_vvc_prof_grad_filter_8x_neon, export=1
        gh              .req x0
        gv              .req x1
        gstride         .req x2
        src             .req x3
        src_stride      .req x4
        width           .req w5
        height          .req w6

        lsl             src_stride, src_stride, #1
        neg             x7, src_stride
1:
        mov             x10, src
        mov             w11, width
        mov             x12, gh
        mov             x13, gv
2:
        ldur            q0, [x10, #2]
        ldur            q1, [x10, #-2]
        subs            w11, w11, #8
        ldr             q2, [x10, src_stride]
        ldr             q3, [x10, x7]
        sshr            v0.8h, v0.8h, #6
        sshr            v1.8h, v1.8h, #6
        sshr            v2.8h, v2.8h, #6
        sshr            v3.8h, v3.8h, #6
        sub             v0.8h, v0.8h, v1.8h
        sub             v2.8h, v2.8h, v3.8h
        st1             {v0.8h}, [x12], #16
        st1             {v2.8h}, [x13], #16
        add             x10, x10, #16
        b.ne            2b

        subs            height, height, #1
        add             gh, gh, gstride, lsl #1
        add             gv, gv, gstride, lsl #1
        add             src, src, src_stride
        b.ne            1b
        ret

.unreq gh
.unreq gv
.unreq gstride
.unreq src
.unreq src_stride
.unreq width
.unreq height

endfunc

.macro vvc_apply_bdof_min_block bit_depth
        dst             .req x0
        dst_stride      .req x1
        src0            .req x2
        src1            .req x3
        gh              .req x4
        gv              .req x5
        vx              .req w6
        vy              .req w7

        dup             v0.4h, vx
        dup             v1.4h, vy
        movi            v7.4s, #(1 << (14 - \bit_depth))
        ldp             x8, x9, [gh]
        ldp             x10, x11, [gv]
        mov             x12, #(BDOF_BLOCK_SIZE * 2)
        mov             w13, #(BDOF_MIN_BLOCK_SIZE)
        mov             x14, #(VVC_MAX_PB_SIZE * 2)
.if \bit_depth >= 10
        // clip pixel
        mov             w15, #((1 << \bit_depth) - 1)
        movi            v18.8h, #0
        lsl             dst_stride, dst_stride, #1
        dup             v17.8h, w15
.endif
1:
        ld1             {v2.4h}, [x8], x12
        ld1             {v3.4h}, [x9], x12
        ld1             {v4.4h}, [x10], x12
        ld1             {v5.4h}, [x11], x12
        sub             v2.4h, v2.4h, v3.4h
        sub             v4.4h, v4.4h, v5.4h
        smull           v2.4s, v0.4h, v2.4h
        smlal           v2.4s, v1.4h, v4.4h

        ld1             {v5.4h}, [src0], x14
        ld1             {v6.4h}, [src1], x14
        saddl           v5.4s, v5.4h, v6.4h
        add             v5.4s, v5.4s, v7.4s
        add             v5.4s, v5.4s, v2.4s
        sqshrn          v5.4h, v5.4s, #(15 - \bit_depth)
        subs            w13, w13, #1
.if \bit_depth == 8
        sqxtun          v5.8b, v5.8h
        str             s5, [dst]
        add             dst, dst, dst_stride
.else
        smin            v5.4h, v5.4h, v17.4h
        smax            v5.4h, v5.4h, v18.4h
        st1             {v5.4h}, [dst], dst_stride
.endif
        b.ne            1b
        ret

.unreq dst
.unreq dst_stride
.unreq src0
.unreq src1
.unreq gh
.unreq gv
.unreq vx
.unreq vy
.endm

function ff_vvc_apply_bdof_min_block_8_neon, export=1
        vvc_apply_bdof_min_block 8
endfunc

function ff_vvc_apply_bdof_min_block_10_neon, export=1
        vvc_apply_bdof_min_block 10
endfunc

function ff_vvc_apply_bdof_min_block_12_neon, export=1
        vvc_apply_bdof_min_block 12
endfunc

.macro derive_bdof_vx_vy_x_begin_end
        ldrh            w19, [x14, x16, lsl #1]     // load from src0
        ldrh            w20, [x15, x16, lsl #1]     // load from src1
        sxth            w19, w19
        sxth            w20, w20
        asr             w19, w19, #4
        asr             w20, w20, #4
        sub             w19, w19, w20               // diff
        add             x17, x16, x13, lsl #4       // idx
        ldrh            w3, [gh0, x17, lsl #1]      // load from gh0
        ldrh            w4, [gh1, x17, lsl #1]      // load from gh1
        sxth            w3, w3
        sxth            w4, w4
        ldrh            w22, [gv0, x17, lsl #1]     // load from gv0
        ldrh            w23, [gv1, x17, lsl #1]     // load from gv1
        add             w3, w3, w4
        asr             w21, w3, #1                 // temph
        sxth            w3, w22
        sxth            w4, w23
        add             w3, w3, w4
        cmp             w21, #0
        asr             w22, w3, #1                 // tempv
        cneg            w20, w21, mi
        csetm           w23, ne
        csinc           w23, w23, wzr, ge           // -VVC_SIGN(temph)
        cmp             w22, #0
        add             sgx2, sgx2, w20
        cneg            w20, w22, mi
        cset            w24, ne
        csinv           w24, w24, wzr, ge           // VVC_SIGN(tempv)
        add             sgy2, sgy2, w20
        madd            sgxgy, w24, w21, sgxgy
        madd            sgxdi, w23, w19, sgxdi
        csetm           w24, ne
        csinc           w24, w24, wzr, ge           // -VVC_SIGN(tempv)
        madd            sgydi, w24, w19, sgydi
.endm

function ff_vvc_derive_bdof_vx_vy_neon, export=1
        src0            .req x0
        src1            .req x1
        pad_mask        .req w2
        gh              .req x3
        gv              .req x4
        gh0             .req x27
        gh1             .req x28
        gv0             .req x25
        gv1             .req x26
        vx              .req x5
        vy              .req x6
        sgx2            .req w7
        sgy2            .req w8
        sgxgy           .req w9
        sgxdi           .req w10
        sgydi           .req w11
        y               .req x12

        stp             x27, x28, [sp, #-80]!
        stp             x25, x26, [sp, #16]
        stp             x23, x24, [sp, #32]
        stp             x21, x22, [sp, #48]
        stp             x19, x20, [sp, #64]

        ldp             gh0, gh1, [gh]
        mov             sgx2, #0
        mov             sgy2, #0
        mov             sgxgy, #0
        mov             sgxdi, #0
        mov             sgydi, #0
        ldp             gv0, gv1, [gv]

        mov             y, #-1
        mov             x13, #-1                    // dy
        tst             pad_mask, #2
        b.eq            1f
        mov             x13, #0                     // dy: pad top
1:
        add             x14, src0, x13, lsl #8      // local src0
        add             x15, src1, x13, lsl #8      // local src1

        // x = -1
        mov             x16, #-1                    // dx
        tst             pad_mask, #1
        b.eq            2f
        mov             x16, #0
2:
        derive_bdof_vx_vy_x_begin_end

        // x = 0 to BDOF_MIN_BLOCK_SIZE - 1
        ldr             d0, [x14]
        ldr             d1, [x15]
        lsl             x19, x13, #5
        ldr             d2, [gh0, x19]
        ldr             d3, [gh1, x19]
        sshr            v0.4h, v0.4h, #4
        sshr            v1.4h, v1.4h, #4
        ssubl           v0.4s, v0.4h, v1.4h         // diff
        ldr             d4, [gv0, x19]
        ldr             d5, [gv1, x19]
        saddl           v2.4s, v2.4h, v3.4h
        saddl           v4.4s, v4.4h, v5.4h
        sshr            v2.4s, v2.4s, #1            // temph
        sshr            v4.4s, v4.4s, #1            // tempv
        abs             v3.4s, v2.4s
        abs             v5.4s, v4.4s
        addv            s3, v3.4s
        addv            s5, v5.4s
        mov             w19, v3.s[0]
        mov             w20, v5.s[0]
        add             sgx2, sgx2, w19
        add             sgy2, sgy2, w20

        movi            v5.4s, #1
        cmgt            v17.4s, v4.4s, #0           // mask > 0
        cmlt            v18.4s, v4.4s, #0           // mask < 0
        and             v17.16b, v17.16b, v5.16b
        and             v18.16b, v18.16b, v5.16b
        neg             v19.4s, v18.4s
        add             v20.4s, v17.4s, v19.4s      // VVC_SIGN(tempv)
        smull           v21.2d, v20.2s, v2.2s
        smlal2          v21.2d, v20.4s, v2.4s
        addp            d21, v21.2d
        mov             w19, v21.s[0]
        add             sgxgy, sgxgy, w19

        smull           v16.2d, v20.2s, v0.2s
        smlal2          v16.2d, v20.4s, v0.4s
        addp            d16, v16.2d
        mov             w19, v16.s[0]
        sub             sgydi, sgydi, w19

        cmgt            v17.4s, v2.4s, #0
        cmlt            v18.4s, v2.4s, #0
        and             v17.16b, v17.16b, v5.16b
        and             v18.16b, v18.16b, v5.16b
        neg             v21.4s, v17.4s
        add             v16.4s, v21.4s, v18.4s      // -VVC_SIGN(temph)
        smull           v20.2d, v16.2s, v0.2s
        smlal2          v20.2d, v16.4s, v0.4s
        addp            d20, v20.2d
        mov             w19, v20.s[0]
        add             sgxdi, sgxdi, w19

        // x = BDOF_MIN_BLOCK_SIZE
        mov             x16, #BDOF_MIN_BLOCK_SIZE   // dx
        tst             pad_mask, #4
        b.eq            3f
        mov             x16, #(BDOF_MIN_BLOCK_SIZE - 1)
3:
        derive_bdof_vx_vy_x_begin_end

        add             y, y, #1
        cmp             y, #(BDOF_MIN_BLOCK_SIZE)
        mov             x13, y
        b.gt            4f
        b.lt            1b
        tst             pad_mask, #8
        b.eq            1b
        sub             x13, x13, #1                // pad bottom
        b               1b
4:
        mov             w3, #31
        mov             w14, #0
        mov             w16, #-15
        mov             w17, #15
        cbz             sgx2, 5f
        clz             w12, sgx2
        lsl             sgxdi, sgxdi, #2
        sub             w13, w3, w12                // log2(sgx2)
        asr             sgxdi, sgxdi, w13
        cmp             sgxdi, w16
        csel            w14, w16, sgxdi, lt         // clip to -15
        b.le            5f
        cmp             sgxdi, w17
        csel            w14, w17, sgxdi, gt         // clip to 15
5:
        str             w14, [vx]

        mov             w15, #0
        cbz             sgy2, 6f
        lsl             sgydi, sgydi, #2
        smull           x14, w14, sgxgy
        asr             w14, w14, #1
        sub             sgydi, sgydi, w14
        clz             w12, sgy2
        sub             w13, w3, w12                // log2(sgy2)
        asr             sgydi, sgydi, w13
        cmp             sgydi, w16
        csel            w15, w16, sgydi, lt         // clip to -15
        b.le            6f
        cmp             sgydi, w17
        csel            w15, w17, sgydi, gt         // clip to 15
6:
        str             w15, [vy]
        ldp             x25, x26, [sp, #16]
        ldp             x23, x24, [sp, #32]
        ldp             x21, x22, [sp, #48]
        ldp             x19, x20, [sp, #64]
        ldp             x27, x28, [sp], #80
        ret
.unreq src0
.unreq src1
.unreq pad_mask
.unreq gh
.unreq gv
.unreq vx
.unreq vy
.unreq sgx2
.unreq sgy2
.unreq sgxgy
.unreq sgxdi
.unreq sgydi
.unreq y
endfunc

