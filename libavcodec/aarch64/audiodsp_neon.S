/*
 * AArch64 NEON optimised audio functions
 * Copyright (c) 2026 FFmpeg
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

// void ff_vector_clipf_neon(float *dst, const float *src, int len,
//                           float min, float max);
// x0 = dst, x1 = src, w2 = len (multiple of 16, > 0), s0 = min, s1 = max
function ff_vector_clipf_neon, export=1
        dup             v16.4s, v0.s[0]         // min broadcast
        dup             v17.4s, v1.s[0]         // max broadcast
1:      ld1             {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
        subs            w2,  w2,  #16
        fmin            v0.4s, v0.4s, v17.4s
        fmin            v1.4s, v1.4s, v17.4s
        fmin            v2.4s, v2.4s, v17.4s
        fmin            v3.4s, v3.4s, v17.4s
        fmax            v0.4s, v0.4s, v16.4s
        fmax            v1.4s, v1.4s, v16.4s
        fmax            v2.4s, v2.4s, v16.4s
        fmax            v3.4s, v3.4s, v16.4s
        st1             {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
        b.gt            1b
        ret
endfunc
