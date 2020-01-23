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

// float ff_compute_distance_ssd_neon(const float *f1, const float *f2, ptrdiff_t len);
function ff_compute_distance_ssd_neon, export=1
	fmov	s0, wzr
	add	x3, x0, x2, lsl #2		// end of f1
	sub	x0, x0, x2, lsl #2		// begin of f1
	sub	x1, x1, x2, lsl #2		// begin of f2
	add	x3, x3, #4			// end + 1 of f1

	// process 32 pairs of data per loop
	add	x4, x0, #128
	cmp	x4, x3
	b.gt	2f
1:	ld1	{v16.4S, v17.4S, v18.4S, v19.4S}, [x0], #64
	ld1	{v20.4S, v21.4S, v22.4S, v23.4S}, [x1], #64
	ld1	{v24.4S, v25.4S, v26.4S, v27.4S}, [x0], #64
	ld1	{v28.4S, v29.4S, v30.4S, v31.4S}, [x1], #64

	fsub	v16.4S, v16.4S, v20.4S

	fsub	v17.4S, v17.4S, v21.4S
	fmul	v16.4S, v16.4S, v16.4S

	fsub	v18.4S, v18.4S, v22.4S
	fmul	v17.4S, v17.4S, v17.4S

	fsub	v19.4S, v19.4S, v23.4S
	fmla	v16.4S, v18.4S, v18.4S

	fsub	v24.4S, v24.4S, v28.4S
	fmla	v17.4S, v19.4S, v19.4S

	fsub	v25.4S, v25.4S, v29.4S
	fmla	v16.4S, v24.4S, v24.4S

	fsub	v26.4S, v26.4S, v30.4S
	fmla	v17.4S, v25.4S, v25.4S

	fsub	v27.4S, v27.4S, v31.4S
	fmla	v16.4S, v26.4S, v26.4S

	fmla	v17.4S, v27.4S, v27.4S

	fadd	v1.4S, v16.4S, v17.4S
	faddp	v1.4S, v1.4S, v1.4S
	faddp	s1, v1.2S
	fadd	s0, s0, s1
	add	x4, x0, #128
	cmp	x4, x3
	b.le	1b

	// process 16 pairs of data per loop
2:	add	x4, x0, #64
	cmp	x4, x3
	b.gt	4f
3:	ld1	{v16.4S, v17.4S, v18.4S, v19.4S}, [x0], #64
	ld1	{v20.4S, v21.4S, v22.4S, v23.4S}, [x1], #64

	fsub	v16.4S, v16.4S, v20.4S

	fsub	v17.4S, v17.4S, v21.4S
	fmul	v16.4S, v16.4S, v16.4S

	fsub	v18.4S, v18.4S, v22.4S
	fmul	v17.4S, v17.4S, v17.4S

	fsub	v19.4S, v19.4S, v23.4S
	fmla	v16.4S, v18.4S, v18.4S

	fmla	v17.4S, v19.4S, v19.4S

	fadd	v1.4S, v16.4S, v17.4S
	faddp	v1.4S, v1.4S, v1.4S
	faddp	s1, v1.2S
	fadd	s0, s0, s1
	add	x4, x0, #64
	cmp	x4, x3
	b.le	3b

	// process 1 pair of data per loop
4:	cmp	x0, x3
	b.eq	6f
5:	ldr	s1, [x0], #4
	ldr	s2, [x1], #4
	fsub	s1, s1, s2
	cmp	x0, x3
	fmadd	s0, s1, s1, s0
	b.ne	5b
6:	ret

endfunc
