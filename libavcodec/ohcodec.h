/*
 * This file is part of FFmpeg.
 *
 * Copyright (c) 2025 Zhao Zhili <quinkblack@foxmail.com>
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

#ifndef AVCODEC_OHCODEC_H
#define AVCODEC_OHCODEC_H

#include <multimedia/player_framework/native_averrors.h>
#include <multimedia/player_framework/native_avformat.h>

#include "libavutil/pixfmt.h"

int ff_oh_err_to_ff_err(OH_AVErrCode err);

enum AVPixelFormat ff_oh_pix_to_ff_pix(OH_AVPixelFormat oh_pix);

#endif
