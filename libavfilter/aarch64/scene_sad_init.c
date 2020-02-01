/*
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

#include "libavutil/aarch64/cpu.h"
#include "libavfilter/scene_sad.h"

void ff_scene_sad_neon(SCENE_SAD_PARAMS);

void ff_scene_sad16_neon(SCENE_SAD_PARAMS);

ff_scene_sad_fn ff_scene_sad_get_fn_aarch64(int depth)
{
    int cpu_flags = av_get_cpu_flags();
    if (have_neon(cpu_flags)) {
        if (depth == 8)
            return ff_scene_sad_neon;
        if (depth == 16)
            return ff_scene_sad16_neon;
    }

    return NULL;
}
