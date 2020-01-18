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

// void ff_fcmul_add_neon(float *sum, const float *t, const float *c, ptrdiff_t len);
function ff_fcmul_add_neon, export=1
1:
	ld2	{v0.4S, v1.4S}, [x1], #32	// load t
	ld2	{v2.4S, v3.4S}, [x2], #32	// load c
	ld2	{v4.4S, v5.4S}, [x0]		// load sum
	fmla	v4.4S, v0.4S, v2.4S
	fmls	v4.4S, v1.4S, v3.4S
	fmla	v5.4S, v0.4S, v3.4S
	fmla	v5.4S, v1.4S, v2.4S
	st2	{v4.4S, v5.4S}, [x0], #32	// store sum
	subs	x3, x3, #4
	b.ne	1b
	ldr	s0, [x1]	// load t
	ldr	s1, [x2]	// load c
	ldr	s2, [x0]	// load sum
	fmul	s0, s0, s1
	fadd	s2, s2, s0
	str	s2, [x0]	// store sum
	ret
endfunc
