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

/*
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
 */

static inline int32x4_t padding_edge(int16x8_t input, uint8x16_t table)
{
    uint8x16_t v0 = vqtbl1q_u8(vreinterpretq_u8_s16(input), table);
    int16x8_t v1 = vreinterpretq_s16_u8(v0);

    int32x4_t low = vaddl_s16(vget_low_s16(v1), vget_low_s16(input));
    int32x4_t high = vaddl_high_s16(v1, input);
    low = vpaddq_s32(low, high);
    return low;
}

static inline int32x4_t padding_edge2(uint16x8_t input, uint8x16_t table)
{
    uint8x16_t v0 = vqtbl1q_u8(vreinterpretq_u8_u16(input), table);
    uint16x8_t v1 = vreinterpretq_u16_u8(v0);

    int32x4_t low = vreinterpretq_s32_u32(vaddl_u16(vget_low_u16(v1), vget_low_u16(input)));
    int32x4_t high = vreinterpretq_s32_u32(vaddl_high_u16(v1, input));
    low = vpaddq_s32(low, high);
    return low;
}

static void vvc_derive_bdof_vx_vy_8x(const int16_t *_src0, const int16_t *_src1, int16_t *gradient_h[2], int16_t *gradient_v[2], int16_t vx[16], int16_t vy[16], int block_h)
{
    int line_table[2][4][5] = {0};
    const int thres = 1 << 4;
    const uint8_t idx[] = {0, 1, 16, 16, 16, 16, 8, 9,
                           6, 7, 16, 16, 16, 16, 14, 15};
    const uint8x16_t edge_table = vld1q_u8(idx);
    for (int y = 0; y < block_h; y += BDOF_MIN_BLOCK_SIZE) {
        const int16_t *src0 = _src0 + y * MAX_PB_SIZE;
        const int16_t *src1 = _src1 + y * MAX_PB_SIZE;
        int idx = BDOF_BLOCK_SIZE * y;
        const int16_t *gh[] = {gradient_h[0] + idx, gradient_h[1] + idx};
        const int16_t *gv[] = {gradient_v[0] + idx, gradient_v[1] + idx};
        const int padTop = !y;
        const int padBottom = y + BDOF_MIN_BLOCK_SIZE == block_h;

        int32x4_t sgx2_v = vdupq_n_s32(0);
        int32x4_t sgy2_v = vdupq_n_s32(0);
        int32x4_t sgxgy_v = vdupq_n_s32(0);
        int32x4_t sgxdi_v = vdupq_n_s32(0);
        int32x4_t sgydi_v = vdupq_n_s32(0);
        for (int y2 = -1; y2 < BDOF_MIN_BLOCK_SIZE + 1; y2++) {
            const int dy = y2 + (padTop && y2 < 0) - (padBottom && y2 == BDOF_MIN_BLOCK_SIZE);
            const int16_t *src02 = src0 + dy * MAX_PB_SIZE;
            const int16_t *src12 = src1 + dy * MAX_PB_SIZE;

            int16x8_t src0_v = vld1q_s16(src02);
            int16x8_t src1_v = vld1q_s16(src12);
            int16x8_t gh0_v = vld1q_s16(&gh[0][BDOF_BLOCK_SIZE * dy]);
            int16x8_t gh1_v = vld1q_s16(&gh[1][BDOF_BLOCK_SIZE * dy]);
            int16x8_t gv0_v = vld1q_s16(&gv[0][BDOF_BLOCK_SIZE * dy]);
            int16x8_t gv1_v = vld1q_s16(&gv[1][BDOF_BLOCK_SIZE * dy]);

            src0_v = vshrq_n_s16(src0_v, 4);
            src1_v = vshrq_n_s16(src1_v, 4);
            // tmph
            gh0_v = vhaddq_s16(gh0_v, gh1_v);
            // tmpv
            gv0_v = vhaddq_s16(gv0_v, gv1_v);
            // diff
            src0_v = vsubq_s16(src0_v, src1_v);

            // vpaddlq_s16
            uint16x8_t abs_temph = vreinterpretq_u16_s16(vabsq_s16(gh0_v));
            uint16x8_t abs_tempv = vreinterpretq_u16_s16(vabsq_s16(gv0_v));
            sgx2_v = vaddq_s32(sgx2_v, padding_edge2(abs_temph, edge_table));
            sgy2_v = vaddq_s32(sgy2_v, padding_edge2(abs_tempv, edge_table));

            int16x8_t sign_temph, sign_tempv;
            {
                uint16x8_t v0 = vcgtq_s16(gh0_v, vdupq_n_s16(0));
                uint16x8_t v1 = vcltq_s16(gh0_v, vdupq_n_s16(0));
                sign_temph = vreinterpretq_s16_u16(vsubq_u16(v1, v0));
            }

            {
                uint16x8_t v0 = vcgtq_s16(gv0_v, vdupq_n_s16(0));
                uint16x8_t v1 = vcltq_s16(gv0_v, vdupq_n_s16(0));
                sign_tempv = vreinterpretq_s16_u16(vsubq_u16(v1, v0));
            }

            {
                int16x8_t v0 = vmulq_s16(sign_tempv, gh0_v);
                int16x8_t v1 = vmulq_s16(sign_temph, src0_v);
                int16x8_t v2 = vmulq_s16(sign_tempv, src0_v);

                sgxgy_v = vaddq_s32(sgxgy_v, padding_edge(v0, edge_table));
                sgxdi_v = vsubq_s32(sgxdi_v, padding_edge(v1, edge_table));
                sgydi_v = vsubq_s32(sgydi_v, padding_edge(v2, edge_table));
            }
        }

        sgx2_v = vpaddq_s32(sgx2_v, sgx2_v);
        sgy2_v = vpaddq_s32(sgy2_v, sgy2_v);
        sgxgy_v = vpaddq_s32(sgxgy_v, sgxgy_v);
        sgxdi_v = vpaddq_s32(sgxdi_v, sgxdi_v);
        sgydi_v = vpaddq_s32(sgydi_v, sgydi_v);

        int32x4_t min = vdupq_n_s32(-15);
        int32x4_t max = vdupq_n_s32(15);

        int32x4_t log2_sgx2 = vsubq_s32(vclzq_s32(sgx2_v), vdupq_n_s32(31));
        sgxdi_v = vshlq_n_s32(sgxdi_v, 2);
        sgxdi_v = vshlq_s32(sgxdi_v, log2_sgx2);
        sgxdi_v = vminq_s32(sgxdi_v, max);
        sgxdi_v = vmaxq_s32(sgxdi_v, min);
        uint32x4_t mask = vcgtq_s32(sgx2_v, vdupq_n_s32(0));
        sgxdi_v = vandq_s32(sgxdi_v, vreinterpretq_s32_u32(mask));
        int16x4_t v1 = vqmovn_s32(sgxdi_v);
        vst1_lane_s32(vx + y, vreinterpret_s32_s16(v1), 0);

        int32x4_t log2_sgy2 = vsubq_s32(vclzq_s32(sgy2_v), vdupq_n_s32(31));
        sgydi_v = vshlq_n_s32(sgydi_v, 2);
        int32x4_t v0 = vmulq_s32(sgxdi_v, sgxgy_v);
        v0 = vshrq_n_s32(v0, 1);
        v0 = vsubq_s32(sgydi_v, v0);
        v0 = vshlq_s32(v0, log2_sgy2);
        v0 = vminq_s32(v0, max);
        v0 = vmaxq_s32(v0, min);
        mask = vcgtq_s32(sgy2_v, vdupq_n_s32(0));
        v0 = vandq_s32(v0, vreinterpretq_s32_u32(mask));
        v1 = vqmovn_s32(v0);
        vst1_lane_s32(vy + y, vreinterpret_s32_s16(v1), 0);
    }
}

static void vvc_derive_bdof_vx_vy_16x(const int16_t *_src0, const int16_t *_src1, int16_t *gradient_h[2], int16_t *gradient_v[2], int16_t vx[16], int16_t vy[16], int block_h)
{
    int line_table[2][4][5] = {0};
    int block_w = 16;
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
            }
            vx[y + x / BDOF_MIN_BLOCK_SIZE] = sgx2 > 0 ? av_clip((sgxdi * (1 << 2)) >> av_log2(sgx2), -thres + 1, thres - 1) : 0;
            vy[y + x / BDOF_MIN_BLOCK_SIZE] = sgy2 > 0 ? av_clip( ((sgydi * (1 << 2)) - ((vx[y + x / BDOF_MIN_BLOCK_SIZE] * sgxgy) >> 1)) >> av_log2(sgy2), -thres + 1, thres - 1) : 0;
        }
    }
}

void ff_vvc_apply_bdof_block_8_neon(uint8_t *dst,
                                    ptrdiff_t dst_stride, const int16_t *src0,
                                    const int16_t *src1,
                                    const int16_t **gh, const int16_t **gv,
                                    int16_t *vx, int16_t *vy);

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

    int16_t vx[16], vy[16];
    if (block_w == 8)
        vvc_derive_bdof_vx_vy_8x(_src0, _src1, gradient_h, gradient_v, vx, vy, block_h);
    else
        vvc_derive_bdof_vx_vy_16x(_src0, _src1, gradient_h, gradient_v, vx, vy, block_h);
    for (int y = 0; y < block_h; y += BDOF_MIN_BLOCK_SIZE) {
        for (int x = 0; x < block_w; x += 2 * BDOF_MIN_BLOCK_SIZE) {
            const int16_t *src0 = _src0 + y * MAX_PB_SIZE + x;
            const int16_t *src1 = _src1 + y * MAX_PB_SIZE + x;
            int idx = BDOF_BLOCK_SIZE * y + x;
            const int16_t *gh[] = {gradient_h[0] + idx, gradient_h[1] + idx};
            const int16_t *gv[] = {gradient_v[0] + idx, gradient_v[1] + idx};
            uint8_t *dst1 = dst + x;;

            int idx1 = y + x / BDOF_MIN_BLOCK_SIZE;
            ff_vvc_apply_bdof_block_8_neon(dst1, dst_stride, src0, src1, gh, gv, vx + idx1, vy + idx1);
        }
        dst += BDOF_MIN_BLOCK_SIZE * dst_stride;
    }
}
