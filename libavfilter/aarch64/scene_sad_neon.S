/*
 * Copyright (c) 2020 Zhao Zhili
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

// void ff_scene_sadx_neon(const uint8_t *src1, ptrdiff_t stride1,
//                         const uint8_t *src2, ptrdiff_t stride2,
//                         ptrdiff_t width, ptrdiff_t height,
//                         uint64_t *sum)
.macro	scene_sad_neon, depth=8
	// x0: src1
	// x1: stride1
	// x2: src2
	// x3: stride2
	// x4: width
	// x5: height
	// x6: sum

	// x7: step of width loop
	// x8: index of row
	// x9: width / x7 * x7
	// x10: sad
	// x11: index of column
	// w12: src1[x]
	// w13: src2[x]

	mov	x8, xzr
	mov	x10, xzr

.if \depth == 8
	mov	x7, #64
	and	x9, x4, #0xFFFFFFFFFFFFFFC0
.endif

.if \depth == 16
	mov	x7, #32
	and	x9, x4, #0xFFFFFFFFFFFFFFE0
.endif

1:	cmp	x4, x7		// check width
	mov	x11, xzr
	b.lt	3f

	mov	v0.d[0], x10

	// vector loop
2:
.if \depth == 8
	add	x14, x0, x11
	add	x15, x2, x11
.endif

.if \depth == 16
	add	x14, x0, x11, lsl #1
	add	x15, x2, x11, lsl #1
.endif
	ld1	{v16.4S, v17.4S, v18.4S, v19.4S}, [x14]
	ld1	{v20.4S, v21.4S, v22.4S, v23.4S}, [x15]
	add	x11, x11, x7
	cmp	x9, x11

.if \depth == 8
	uabd	v16.16B, v16.16B, v20.16B
	uabd	v17.16B, v17.16B, v21.16B
	uabd	v18.16B, v18.16B, v22.16B
	uabd	v19.16B, v19.16B, v23.16B
	uaddlv	h16, v16.16B
	uaddlv	h17, v17.16B
	uaddlv	h18, v18.16B
	uaddlv	h19, v19.16B
.endif

.if \depth == 16
	uabd	v16.8H, v16.8H, v20.8H
	uabd	v17.8H, v17.8H, v21.8H
	uabd	v18.8H, v18.8H, v22.8H
	uabd	v19.8H, v19.8H, v23.8H
	uaddlv	s16, v16.8H
	uaddlv	s17, v17.8H
	uaddlv	s18, v18.8H
	uaddlv	s19, v19.8H
.endif

	add	d16, d16, d17
	add	d18, d18, d19
	add	d0, d0, d16
	add	d0, d0, d18

	b.ne	2b

	cmp	x9, x4
	fmov	x10, d0
	b.eq	4f

	// scalar loop
3:
.if \depth == 8
	ldrb	w12, [x0, x11]
	ldrb	w13, [x2, x11]
.endif

.if \depth == 16
	ldrh	w12, [x0, x11, lsl #1]
	ldrh	w13, [x2, x11, lsl #1]
.endif
	add	x11, x11, #1
	subs	w12, w12, w13
	cneg	w12, w12, mi
	add	x10, x10, x12
	cmp	x11, x4
	b.ne	3b

	// next row
4:
	add	x8, x8, #1
	add	x0, x0, x1
	cmp	x8, x5
	add	x2, x2, x3
	b.ne	1b

5:
	str	x10, [x6]
	ret
.endm

function ff_scene_sad_neon, export=1
	scene_sad_neon	depth=8
endfunc

function ff_scene_sad16_neon, export=1
	scene_sad_neon	depth=16
endfunc
