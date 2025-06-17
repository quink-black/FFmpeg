#include <arm_neon.h>

#include "libavutil/common.h"
#include "libavcodec/vvc/ctu.h"

#include "of.h"

#define BDOF_BLOCK_SIZE         16
#define BDOF_MIN_BLOCK_SIZE     4

static void prof_grad_filter(int16_t *gradient_h, int16_t *gradient_v,
                             const ptrdiff_t gradient_stride,
                             const int16_t *_src, const ptrdiff_t src_stride,
                             const int width, const int height) {
    const int shift = 6;
    const int16_t *src = _src;

    for (int y = 0; y < height; y++) {
        const int16_t *p = src;
        for (int x = 0; x < width; x++) {
            gradient_h[x] = (p[1] >> shift) - (p[-1] >> shift);
            gradient_v[x] = (p[src_stride] >> shift) - (p[-src_stride] >> shift);
            p++;
        }
        gradient_h += gradient_stride;
        gradient_v += gradient_stride;
        src += src_stride;
    }
}

void ff_vvc_prof_grad_filter_8x_neon(int16_t *gradient_h,
                                     int16_t *gradient_v,
                                     ptrdiff_t gradient_stride,
                                     const int16_t *_src,
                                     ptrdiff_t src_stride,
                                     int width, int height);

#define VVC_SIGN(v) (v < 0 ? -1 : !!v)

static void vvc_derive_bdof_vx_vy(const int16_t *_src0, const int16_t *_src1, int16_t *gradient_h[2], int16_t *gradient_v[2], int vx[16], int vy[16], int block_w, int block_h)
{
    int line_table[2][4][5] = {0};
    for (int y = 0; y < block_h; y += BDOF_MIN_BLOCK_SIZE) {
        for (int x = 0; x < block_w; x += BDOF_MIN_BLOCK_SIZE) {
            const int16_t *src0 = _src0 + y * MAX_PB_SIZE + x;
            const int16_t *src1 = _src1 + y * MAX_PB_SIZE + x;
            int idx = BDOF_BLOCK_SIZE * y + x;
            const int16_t *gh[] = {gradient_h[0] + idx, gradient_h[1] + idx};
            const int16_t *gv[] = {gradient_v[0] + idx, gradient_v[1] + idx};
            const int shift2 = 4;
            const int shift3 = 1;
            const int thres = 1 << 4;
            int sgx2 = 0, sgy2 = 0, sgxgy = 0, sgxdi = 0, sgydi = 0;
            const int padLeft = !x;
            const int padTop = !y;
            const int padRight = x + BDOF_MIN_BLOCK_SIZE == block_w;
            const int padBottom = y + BDOF_MIN_BLOCK_SIZE == block_h;

            for (int y2 = -1; y2 < BDOF_MIN_BLOCK_SIZE + 1; y2++) {
                if (y == 0) {
                    // pad top, copy last line
                    if (y2 == 0) {
                        sgx2 += line_table[0][x / BDOF_MIN_BLOCK_SIZE][0];
                        sgy2 += line_table[0][x / BDOF_MIN_BLOCK_SIZE][1];
                        sgxgy += line_table[0][x / BDOF_MIN_BLOCK_SIZE][2];
                        sgxdi += line_table[0][x / BDOF_MIN_BLOCK_SIZE][3];
                        sgydi += line_table[0][x / BDOF_MIN_BLOCK_SIZE][4];
                        continue;
                    }
                } else if (y > 0) {
                    if (y2 == -1) {
                        sgx2 += line_table[0][x / BDOF_MIN_BLOCK_SIZE][0];
                        sgy2 += line_table[0][x / BDOF_MIN_BLOCK_SIZE][1];
                        sgxgy += line_table[0][x / BDOF_MIN_BLOCK_SIZE][2];
                        sgxdi += line_table[0][x / BDOF_MIN_BLOCK_SIZE][3];
                        sgydi += line_table[0][x / BDOF_MIN_BLOCK_SIZE][4];
                        continue;
                    } else if (y2 == 0) {
                        sgx2 += line_table[1][x / BDOF_MIN_BLOCK_SIZE][0];
                        sgy2 += line_table[1][x / BDOF_MIN_BLOCK_SIZE][1];
                        sgxgy += line_table[1][x / BDOF_MIN_BLOCK_SIZE][2];
                        sgxdi += line_table[1][x / BDOF_MIN_BLOCK_SIZE][3];
                        sgydi += line_table[1][x / BDOF_MIN_BLOCK_SIZE][4];
                        continue;
                    }

                    // pad bottom, copy last line
                    if (y + BDOF_MIN_BLOCK_SIZE == block_h && y2 == BDOF_MIN_BLOCK_SIZE) {
                        sgx2 += line_table[0][x / BDOF_MIN_BLOCK_SIZE][0];
                        sgy2 += line_table[0][x / BDOF_MIN_BLOCK_SIZE][1];
                        sgxgy += line_table[0][x / BDOF_MIN_BLOCK_SIZE][2];
                        sgxdi += line_table[0][x / BDOF_MIN_BLOCK_SIZE][3];
                        sgydi += line_table[0][x / BDOF_MIN_BLOCK_SIZE][4];
                        break;
                    }
                }

                const int dy = y2 + (padTop && y2 < 0) - (padBottom && y2 == BDOF_MIN_BLOCK_SIZE);
                const int16_t *src02 = src0 + dy * MAX_PB_SIZE;
                const int16_t *src12 = src1 + dy * MAX_PB_SIZE;

                int save[5] = {0};
                for (int x1 = -1; x1 < BDOF_MIN_BLOCK_SIZE + 1; x1++) {
                    const int dx = x1 + (padLeft && x1 < 0) - (padRight && x1 == BDOF_MIN_BLOCK_SIZE);
                    const int diff = (src02[dx] >> shift2) - (src12[dx] >> shift2);
                    const int idx1 = BDOF_BLOCK_SIZE * dy + dx;
                    const int temph = (gh[0][idx1] + gh[1][idx1]) >> shift3;
                    const int tempv = (gv[0][idx1] + gv[1][idx1]) >> shift3;

                    save[0] += FFABS(temph);
                    save[1] += FFABS(tempv);
                    save[2] += VVC_SIGN(tempv) * temph;
                    save[3] += -VVC_SIGN(temph) * diff;
                    save[4] += -VVC_SIGN(tempv) * diff;
                }
                sgx2 += save[0];
                sgy2 += save[1];
                sgxgy += save[2];
                sgxdi += save[3];
                sgydi += save[4];

                if (y2 == BDOF_MIN_BLOCK_SIZE - 1 || (y == 0 && y2 == -1)) {
                    line_table[0][x / BDOF_MIN_BLOCK_SIZE][0] = save[0];
                    line_table[0][x / BDOF_MIN_BLOCK_SIZE][1] = save[1];
                    line_table[0][x / BDOF_MIN_BLOCK_SIZE][2] = save[2];
                    line_table[0][x / BDOF_MIN_BLOCK_SIZE][3] = save[3];
                    line_table[0][x / BDOF_MIN_BLOCK_SIZE][4] = save[4];
                } else if (y2 == BDOF_MIN_BLOCK_SIZE) {
                    line_table[1][x / BDOF_MIN_BLOCK_SIZE][0] = save[0];
                    line_table[1][x / BDOF_MIN_BLOCK_SIZE][1] = save[1];
                    line_table[1][x / BDOF_MIN_BLOCK_SIZE][2] = save[2];
                    line_table[1][x / BDOF_MIN_BLOCK_SIZE][3] = save[3];
                    line_table[1][x / BDOF_MIN_BLOCK_SIZE][4] = save[4];
                }
            }
            vx[y + x / BDOF_MIN_BLOCK_SIZE] = sgx2 > 0 ? av_clip((sgxdi * (1 << 2)) >> av_log2(sgx2), -thres + 1, thres - 1) : 0;
            vy[y + x / BDOF_MIN_BLOCK_SIZE] = sgy2 > 0 ? av_clip( ((sgydi * (1 << 2)) - ((vx[y + x / BDOF_MIN_BLOCK_SIZE] * sgxgy) >> 1)) >> av_log2(sgy2), -thres + 1, thres - 1) : 0;
        }
    }
}

void ff_apply_bdof_intrinsic(uint8_t *_dst, ptrdiff_t _dst_stride,
                               const int16_t *_src0, const int16_t *_src1,
                               int block_w, int block_h)
{
    // +2 for pad left and right
    int16_t gradient_buf_h[2][BDOF_BLOCK_SIZE * BDOF_BLOCK_SIZE + 2];
    int16_t gradient_buf_v[2][BDOF_BLOCK_SIZE * BDOF_BLOCK_SIZE + 2];
    int16_t *gradient_h[2] = {&gradient_buf_h[0][1], &gradient_buf_h[1][1]};
    int16_t *gradient_v[2] = {&gradient_buf_v[0][1], &gradient_buf_v[1][1]};
    ptrdiff_t dst_stride = _dst_stride;
    uint8_t *dst = _dst;

    ff_vvc_prof_grad_filter_8x_neon(gradient_h[0], gradient_v[0],
                     BDOF_BLOCK_SIZE,
                     _src0, MAX_PB_SIZE, block_w, block_h);
    ff_vvc_prof_grad_filter_8x_neon(gradient_h[1], gradient_v[1],
                     BDOF_BLOCK_SIZE,
                     _src1, MAX_PB_SIZE, block_w, block_h);

    int vx_buf[16], vy_buf[16];
    vvc_derive_bdof_vx_vy(_src0, _src1, gradient_h, gradient_v, vx_buf, vy_buf, block_w, block_h);
    for (int y = 0; y < block_h; y += BDOF_MIN_BLOCK_SIZE) {
        for (int x = 0; x < block_w; x += BDOF_MIN_BLOCK_SIZE) {
            const int16_t *src0 = _src0 + y * MAX_PB_SIZE + x;
            const int16_t *src1 = _src1 + y * MAX_PB_SIZE + x;
            int idx = BDOF_BLOCK_SIZE * y + x;
            const int16_t *gh[] = {gradient_h[0] + idx, gradient_h[1] + idx};
            const int16_t *gv[] = {gradient_v[0] + idx, gradient_v[1] + idx};
            uint8_t *dst1 = dst + x;;
            const int16_t *src01 = src0;
            const int16_t *src11 = src1;
            const int bitDepth = 8;
            const int shift4 = 15 - bitDepth;
            const int offset4 = 1 << (shift4 - 1);

            int vx = vx_buf[y + x / BDOF_MIN_BLOCK_SIZE];
            int vy = vy_buf[y + x / BDOF_MIN_BLOCK_SIZE];
            for (int y3 = 0; y3 < BDOF_MIN_BLOCK_SIZE; y3++) {
                for (int x2 = 0; x2 < BDOF_MIN_BLOCK_SIZE; x2++) {
                    const int idx2 = y3 * BDOF_BLOCK_SIZE + x2;
                    const int bdofOffset = vx * (gh[0][idx2] - gh[1][idx2]) +
                                           vy * (gv[0][idx2] - gv[1][idx2]);
                    dst1[x2] = av_clip_uint8((src01[x2] + offset4 + src11[x2] + bdofOffset) >> shift4);
                }
                dst1 += dst_stride;
                src01 += MAX_PB_SIZE;
                src11 += MAX_PB_SIZE;
            }
        }
        dst += BDOF_MIN_BLOCK_SIZE * dst_stride;
    }
}