#include <arm_neon.h>

#include "libavutil/common.h"
#include "libavcodec/vvc/ctu.h"

#include "of.h"

#define BDOF_BLOCK_SIZE         16
#define BDOF_MIN_BLOCK_SIZE     4

static inline int32x4_t padding_edge(int16x8_t input, uint8x16_t table)
{
    uint8x16_t v0 = vqtbl1q_u8(vreinterpretq_u8_s16(input), table);
    int16x8_t v1 = vreinterpretq_s16_u8(v0);

    int32x4_t low = vaddl_s16(vget_low_s16(v1), vget_low_s16(input));
    int32x4_t high = vaddl_high_s16(v1, input);
    low = vpaddq_s32(low, high);
    return low;
}

static inline int32x4_t padding_edge16(int16x8x2_t input, uint8x16x2_t table)
{
    uint8x16x2_t _input = {
            vreinterpretq_u8_s16(input.val[0]),
            vreinterpretq_u8_s16(input.val[1]),
    };
    uint8x16_t v0 = vqtbl2q_u8(_input, table.val[0]);
    int16x8_t v1 = vreinterpretq_s16_u8(v0);

    int32x4_t low = vaddl_s16(vget_low_s16(v1), vget_low_s16(input.val[0]));
    int32x4_t high = vaddl_high_s16(v1, input.val[0]);
    low = vpaddq_s32(low, high);

    v0 = vqtbl2q_u8(_input, table.val[1]);
    v1 = vreinterpretq_s16_u8(v0);
    int32x4_t low1 = vaddl_s16(vget_low_s16(v1), vget_low_s16(input.val[1]));
    int32x4_t high1 = vaddl_high_s16(v1, input.val[1]);
    low1 = vpaddq_s32(low1, high1);

    low = vpaddq_s32(low, low1);
    return low;
}

void ff_vvc_derive_bdof_vx_vy_8x_intrinsic(const int16_t *_src0, const int16_t *_src1, int16_t *gradient_h[2], int16_t *gradient_v[2], int16_t vx[16], int16_t vy[16], int block_h)
{
    int line_table[2][5][4] = {0};
    const uint8_t idx[] = {0, 1, 16, 16, 16, 16, 8, 9,
                           6, 7, 16, 16, 16, 16, 14, 15};
    const uint8x16_t edge_table = vld1q_u8(idx);
    for (int y = 0; y < block_h; y += BDOF_MIN_BLOCK_SIZE) {
        const int16_t *src0 = _src0 + y * MAX_PB_SIZE;
        const int16_t *src1 = _src1 + y * MAX_PB_SIZE;
        int idx = BDOF_BLOCK_SIZE * y;
        const int16_t *gh[] = {gradient_h[0] + idx, gradient_h[1] + idx};
        const int16_t *gv[] = {gradient_v[0] + idx, gradient_v[1] + idx};
        const int padBottom = y + BDOF_MIN_BLOCK_SIZE == block_h;

        int32x4_t sgx2_v = vdupq_n_s32(0);
        int32x4_t sgy2_v = vdupq_n_s32(0);
        int32x4_t sgxgy_v = vdupq_n_s32(0);
        int32x4_t sgxdi_v = vdupq_n_s32(0);
        int32x4_t sgydi_v = vdupq_n_s32(0);
        for (int y2 = 0; y2 < BDOF_MIN_BLOCK_SIZE + 1; y2++) {
            int32x4_t p0, p1, p2, p3, p4;

            if (y > 0 && y2 == 0) {
                p0 = vld1q_s32(&line_table[0][0][0]);
                p1 = vld1q_s32(&line_table[0][1][0]);
                p2 = vld1q_s32(&line_table[0][2][0]);
                p3 = vld1q_s32(&line_table[0][3][0]);
                p4 = vld1q_s32(&line_table[0][4][0]);
                sgx2_v = vaddq_s32(sgx2_v, p0);
                sgy2_v = vaddq_s32(sgy2_v, p1);
                sgxgy_v = vaddq_s32(sgxgy_v, p2);
                sgxdi_v = vsubq_s32(sgxdi_v, p3);
                sgydi_v = vsubq_s32(sgydi_v, p4);
                p0 = vld1q_s32(&line_table[1][0][0]);
                p1 = vld1q_s32(&line_table[1][1][0]);
                p2 = vld1q_s32(&line_table[1][2][0]);
                p3 = vld1q_s32(&line_table[1][3][0]);
                p4 = vld1q_s32(&line_table[1][4][0]);
                sgx2_v = vaddq_s32(sgx2_v, p0);
                sgy2_v = vaddq_s32(sgy2_v, p1);
                sgxgy_v = vaddq_s32(sgxgy_v, p2);
                sgxdi_v = vsubq_s32(sgxdi_v, p3);
                sgydi_v = vsubq_s32(sgydi_v, p4);
                continue;
            }
            const int dy = y2;
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
            int16x8_t abs_temph = vabsq_s16(gh0_v);
            int16x8_t abs_tempv = vabsq_s16(gv0_v);
            p0 = padding_edge(abs_temph, edge_table);
            p1 = padding_edge(abs_tempv, edge_table);

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
                p2 = padding_edge(v0, edge_table);
                p3 = padding_edge(v1, edge_table);
                p4 = padding_edge(v2, edge_table);
            }

            sgx2_v = vaddq_s32(sgx2_v, p0);
            sgy2_v = vaddq_s32(sgy2_v, p1);
            sgxgy_v = vaddq_s32(sgxgy_v, p2);
            sgxdi_v = vsubq_s32(sgxdi_v, p3);
            sgydi_v = vsubq_s32(sgydi_v, p4);
            if (y == 0 && y2 == 0) {
                sgx2_v = vaddq_s32(sgx2_v, p0);
                sgy2_v = vaddq_s32(sgy2_v, p1);
                sgxgy_v = vaddq_s32(sgxgy_v, p2);
                sgxdi_v = vsubq_s32(sgxdi_v, p3);
                sgydi_v = vsubq_s32(sgydi_v, p4);
            }

            if (y2 == BDOF_MIN_BLOCK_SIZE - 1) {
                if (padBottom) {
                    sgx2_v = vaddq_s32(sgx2_v, p0);
                    sgy2_v = vaddq_s32(sgy2_v, p1);
                    sgxgy_v = vaddq_s32(sgxgy_v, p2);
                    sgxdi_v = vsubq_s32(sgxdi_v, p3);
                    sgydi_v = vsubq_s32(sgydi_v, p4);
                    break;
                }
                vst1q_s32(&line_table[0][0][0], p0);
                vst1q_s32(&line_table[0][1][0], p1);
                vst1q_s32(&line_table[0][2][0], p2);
                vst1q_s32(&line_table[0][3][0], p3);
                vst1q_s32(&line_table[0][4][0], p4);
            } else if (y2 == BDOF_MIN_BLOCK_SIZE) {
                vst1q_s32(&line_table[1][0][0], p0);
                vst1q_s32(&line_table[1][1][0], p1);
                vst1q_s32(&line_table[1][2][0], p2);
                vst1q_s32(&line_table[1][3][0], p3);
                vst1q_s32(&line_table[1][4][0], p4);
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

void ff_vvc_derive_bdof_vx_vy_16x_intrinsic(const int16_t *_src0,
                                            const int16_t *_src1,
                                            int16_t *gradient_h[2],
                                            int16_t *gradient_v[2],
                                            int16_t vx[16], int16_t vy[16],
                                            int block_h)
{
    int line_table[2][5][4] = {0};
    const uint8_t idx[] = {0,  1,  64, 64, 64, 64, 8,  9,
                           6,  7,  64, 64, 64, 64, 16, 17,
                           14, 15, 64, 64, 64, 64, 24, 25,
                           22, 23, 64, 64, 64, 64, 30, 31};
    const uint8x16x2_t edge_table = vld1q_u8_x2(idx);
    for (int y = 0; y < block_h; y += BDOF_MIN_BLOCK_SIZE) {
        const int16_t *src0 = _src0 + y * MAX_PB_SIZE;
        const int16_t *src1 = _src1 + y * MAX_PB_SIZE;
        int idx = BDOF_BLOCK_SIZE * y;
        const int16_t *gh[] = {gradient_h[0] + idx, gradient_h[1] + idx};
        const int16_t *gv[] = {gradient_v[0] + idx, gradient_v[1] + idx};
        const int padBottom = y + BDOF_MIN_BLOCK_SIZE == block_h;

        int32x4_t sgx2_v = vdupq_n_s32(0);
        int32x4_t sgy2_v = vdupq_n_s32(0);
        int32x4_t sgxgy_v = vdupq_n_s32(0);
        int32x4_t sgxdi_v = vdupq_n_s32(0);
        int32x4_t sgydi_v = vdupq_n_s32(0);
        for (int y2 = 0; y2 < BDOF_MIN_BLOCK_SIZE + 1; y2++) {
            int32x4_t p0, p1, p2, p3, p4;

            if (y > 0 && y2 == 0) {
                p0 = vld1q_s32(&line_table[0][0][0]);
                p1 = vld1q_s32(&line_table[0][1][0]);
                p2 = vld1q_s32(&line_table[0][2][0]);
                p3 = vld1q_s32(&line_table[0][3][0]);
                p4 = vld1q_s32(&line_table[0][4][0]);
                sgx2_v = vaddq_s32(sgx2_v, p0);
                sgy2_v = vaddq_s32(sgy2_v, p1);
                sgxgy_v = vaddq_s32(sgxgy_v, p2);
                sgxdi_v = vsubq_s32(sgxdi_v, p3);
                sgydi_v = vsubq_s32(sgydi_v, p4);
                p0 = vld1q_s32(&line_table[1][0][0]);
                p1 = vld1q_s32(&line_table[1][1][0]);
                p2 = vld1q_s32(&line_table[1][2][0]);
                p3 = vld1q_s32(&line_table[1][3][0]);
                p4 = vld1q_s32(&line_table[1][4][0]);
                sgx2_v = vaddq_s32(sgx2_v, p0);
                sgy2_v = vaddq_s32(sgy2_v, p1);
                sgxgy_v = vaddq_s32(sgxgy_v, p2);
                sgxdi_v = vsubq_s32(sgxdi_v, p3);
                sgydi_v = vsubq_s32(sgydi_v, p4);
                continue;
            }
            const int dy = y2;
            const int16_t *src02 = src0 + dy * MAX_PB_SIZE;
            const int16_t *src12 = src1 + dy * MAX_PB_SIZE;

            int16x8x2_t src0_v = vld1q_s16_x2(src02);
            int16x8x2_t src1_v = vld1q_s16_x2(src12);
            int16x8x2_t gh0_v = vld1q_s16_x2(&gh[0][BDOF_BLOCK_SIZE * dy]);
            int16x8x2_t gh1_v = vld1q_s16_x2(&gh[1][BDOF_BLOCK_SIZE * dy]);
            int16x8x2_t gv0_v = vld1q_s16_x2(&gv[0][BDOF_BLOCK_SIZE * dy]);
            int16x8x2_t gv1_v = vld1q_s16_x2(&gv[1][BDOF_BLOCK_SIZE * dy]);

            src0_v.val[0] = vshrq_n_s16(src0_v.val[0], 4);
            src1_v.val[0] = vshrq_n_s16(src1_v.val[0], 4);
            src0_v.val[1] = vshrq_n_s16(src0_v.val[1], 4);
            src1_v.val[1] = vshrq_n_s16(src1_v.val[1], 4);
            // tmph
            gh0_v.val[0] = vhaddq_s16(gh0_v.val[0], gh1_v.val[0]);
            // tmpv
            gv0_v.val[0] = vhaddq_s16(gv0_v.val[0], gv1_v.val[0]);
            // diff
            src0_v.val[0] = vsubq_s16(src0_v.val[0], src1_v.val[0]);

            // tmph
            gh0_v.val[1] = vhaddq_s16(gh0_v.val[1], gh1_v.val[1]);
            // tmpv
            gv0_v.val[1] = vhaddq_s16(gv0_v.val[1], gv1_v.val[1]);
            // diff
            src0_v.val[1] = vsubq_s16(src0_v.val[1], src1_v.val[1]);

            int16x8x2_t abs_temph = {vabsq_s16(gh0_v.val[0]), vabsq_s16(gh0_v.val[1])};
            int16x8x2_t abs_tempv = {vabsq_s16(gv0_v.val[0]), vabsq_s16(gv0_v.val[1])};
            p0 = padding_edge16(abs_temph, edge_table);
            p1 = padding_edge16(abs_tempv, edge_table);

            int16x8_t sign_temph[2], sign_tempv[2];
            {
                uint16x8_t v0 = vcgtq_s16(gh0_v.val[0], vdupq_n_s16(0));
                uint16x8_t v1 = vcltq_s16(gh0_v.val[0], vdupq_n_s16(0));
                sign_temph[0] = vreinterpretq_s16_u16(vsubq_u16(v1, v0));

                v0 = vcgtq_s16(gh0_v.val[1], vdupq_n_s16(0));
                v1 = vcltq_s16(gh0_v.val[1], vdupq_n_s16(0));
                sign_temph[1] = vreinterpretq_s16_u16(vsubq_u16(v1, v0));

                v0 = vcgtq_s16(gv0_v.val[0], vdupq_n_s16(0));
                v1 = vcltq_s16(gv0_v.val[0], vdupq_n_s16(0));
                sign_tempv[0] = vreinterpretq_s16_u16(vsubq_u16(v1, v0));

                v0 = vcgtq_s16(gv0_v.val[1], vdupq_n_s16(0));
                v1 = vcltq_s16(gv0_v.val[1], vdupq_n_s16(0));
                sign_tempv[1] = vreinterpretq_s16_u16(vsubq_u16(v1, v0));
            }

            {
                int16x8x2_t v0 = {vmulq_s16(sign_tempv[0], gh0_v.val[0]), vmulq_s16(sign_tempv[1], gh0_v.val[1])};
                int16x8x2_t v1 = {vmulq_s16(sign_temph[0], src0_v.val[0]), vmulq_s16(sign_temph[1], src0_v.val[1])};
                int16x8x2_t v2 = {vmulq_s16(sign_tempv[0], src0_v.val[0]), vmulq_s16(sign_tempv[1], src0_v.val[1])};
                p2 = padding_edge16(v0, edge_table);
                p3 = padding_edge16(v1, edge_table);
                p4 = padding_edge16(v2, edge_table);
            }

            sgx2_v = vaddq_s32(sgx2_v, p0);
            sgy2_v = vaddq_s32(sgy2_v, p1);
            sgxgy_v = vaddq_s32(sgxgy_v, p2);
            sgxdi_v = vsubq_s32(sgxdi_v, p3);
            sgydi_v = vsubq_s32(sgydi_v, p4);
            if (y == 0 && y2 == 0) {
                sgx2_v = vaddq_s32(sgx2_v, p0);
                sgy2_v = vaddq_s32(sgy2_v, p1);
                sgxgy_v = vaddq_s32(sgxgy_v, p2);
                sgxdi_v = vsubq_s32(sgxdi_v, p3);
                sgydi_v = vsubq_s32(sgydi_v, p4);
            }

            if (y2 == BDOF_MIN_BLOCK_SIZE - 1) {
                if (padBottom) {
                    sgx2_v = vaddq_s32(sgx2_v, p0);
                    sgy2_v = vaddq_s32(sgy2_v, p1);
                    sgxgy_v = vaddq_s32(sgxgy_v, p2);
                    sgxdi_v = vsubq_s32(sgxdi_v, p3);
                    sgydi_v = vsubq_s32(sgydi_v, p4);
                    break;
                }
                vst1q_s32(&line_table[0][0][0], p0);
                vst1q_s32(&line_table[0][1][0], p1);
                vst1q_s32(&line_table[0][2][0], p2);
                vst1q_s32(&line_table[0][3][0], p3);
                vst1q_s32(&line_table[0][4][0], p4);
            } else if (y2 == BDOF_MIN_BLOCK_SIZE) {
                vst1q_s32(&line_table[1][0][0], p0);
                vst1q_s32(&line_table[1][1][0], p1);
                vst1q_s32(&line_table[1][2][0], p2);
                vst1q_s32(&line_table[1][3][0], p3);
                vst1q_s32(&line_table[1][4][0], p4);
            }
        }

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
        vst1_lane_u64(vx + y, vreinterpret_u64_s16(v1), 0);

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
        vst1_lane_u64(vy + y, vreinterpret_u64_s16(v1), 0);
    }
}