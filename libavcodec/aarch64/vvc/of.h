//
// Created by quink on 2025/6/17.
//

#ifndef FFMPEG_OF_H
#define FFMPEG_OF_H

#include "stdint.h"

void ff_apply_bdof_intrinsic(uint8_t *_dst, ptrdiff_t _dst_stride,
                               const int16_t *_src0, const int16_t *_src1,
                               int block_w, int block_h);
#endif //FFMPEG_OF_H
