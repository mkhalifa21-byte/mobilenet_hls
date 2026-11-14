/*
 * mobilenet_kernels.cpp  --  BASELINE (naive) HLS kernels
 *
 * These are deliberately unoptimized: no PIPELINE / UNROLL / ARRAY_PARTITION,
 * and every read/write goes straight to DDR through m_axi. Synthesize this
 * version first and record latency + resources: that is the "before" column
 * of your results table. Optimized versions replace these one at a time and
 * must still pass the testbench bit-exactly.
 *
 * Loop order follows the Lecture 2 loop-tiling slides:
 *   row -> col -> output channel -> input channel -> i -> j
 *
 * LOOP_TRIPCOUNT only tells the HLS report how many iterations to assume for
 * variable-bound loops (it does not change the hardware).
 */
#include "mobilenet.h"

/* ------------------------------------------------------------------------
 * Standard 3x3 convolution (only used by layer Conv2d_0: 3 -> 16 channels,
 * 160x160 -> 80x80)
 *   in  [N][H][H]   out [M][Ho][Ho]   w [M][N][3][3]   b [M]
 * ------------------------------------------------------------------------ */
void conv3x3(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
             int N, int M, int H, int S, const int8_t *shift)
{
    const int Ho  = out_size(H, S);
    const int pad = same_pad_begin(H, S, 3);

ROW: for (int row = 0; row < Ho; row++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
    COL: for (int col = 0; col < Ho; col++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
        TO: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            acc_t acc = b[to];
            TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
                KI: for (int i = 0; i < 3; i++) {
                    KJ: for (int j = 0; j < 3; j++) {
                        int y = S * row + i - pad;
                        int x = S * col + j - pad;
                        if (y >= 0 && y < H && x >= 0 && x < H)
                            acc += (acc_t)in[(ti * H + y) * H + x] *
                                   (acc_t)w[((to * N + ti) * 3 + i) * 3 + j];
                    }
                }
            }
            out[(to * Ho + row) * Ho + col] = requant(acc, shift[to], 1);
        }
    }
}
}

/* ------------------------------------------------------------------------
 * Depthwise 3x3 convolution: channel c of the output only looks at channel c
 * of the input (slim.separable_conv2d with num_outputs=None).
 *   in [C][H][H]   out [C][Ho][Ho]   w [C][3][3]   b [C]
 * ------------------------------------------------------------------------ */
void dwconv3x3(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
               int C, int H, int S, const int8_t *shift)
{
    const int Ho  = out_size(H, S);
    const int pad = same_pad_begin(H, S, 3);

ROW: for (int row = 0; row < Ho; row++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
    COL: for (int col = 0; col < Ho; col++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
        CH: for (int c = 0; c < C; c++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512
            acc_t acc = b[c];
            KI: for (int i = 0; i < 3; i++) {
                KJ: for (int j = 0; j < 3; j++) {
                    int y = S * row + i - pad;
                    int x = S * col + j - pad;
                    if (y >= 0 && y < H && x >= 0 && x < H)
                        acc += (acc_t)in[(c * H + y) * H + x] *
                               (acc_t)w[(c * 3 + i) * 3 + j];
                }
            }
            out[(c * Ho + row) * Ho + col] = requant(acc, shift[c], 1);
        }
    }
}
}

/* ------------------------------------------------------------------------
 * Pointwise 1x1 convolution: at every pixel, out[to] = sum_ti w[to][ti]*in[ti]
 *   in [N][H][H]   out [M][H][H]   w [M][N]   b [M]
 * ------------------------------------------------------------------------ */
void conv1x1(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
             int N, int M, int H, const int8_t *shift)
{
    const int HW = H * H;

ROW: for (int row = 0; row < H; row++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
    COL: for (int col = 0; col < H; col++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
        TO: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=512
            acc_t acc = b[to];
            TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512
                acc += (acc_t)in[ti * HW + row * H + col] *
                       (acc_t)w[to * N + ti];
            }
            out[to * HW + row * H + col] = requant(acc, shift[to], 1);
        }
    }
}
}

/* ------------------------------------------------------------------------
 * Global average pool: [C][H][H] -> [C]   (AvgPool_1a: 5x5 at 160 input,
 * 7x7 at 224 input; slim shrinks the 7x7 kernel to the map size)
 * Inputs are >= 0 (after ReLU6), so (sum + HW/2) / HW rounds to nearest.
 * ------------------------------------------------------------------------ */
void avgpool(const data_t *in, data_t *out, int C, int H)
{
    const int HW = H * H;
CH: for (int c = 0; c < C; c++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
        acc_t sum = 0;
    PIX: for (int p = 0; p < HW; p++) {
#pragma HLS LOOP_TRIPCOUNT min=25 max=25
            sum += in[c * HW + p];
        }
        out[c] = (data_t)((sum + HW / 2) / HW);
    }
}

/* ------------------------------------------------------------------------
 * Classifier (Conv2d_1c_1x1): mathematically a 1x1 conv on a 1x1 map.
 * Kept separate here only because it outputs raw int32 scores (no ReLU6,
 * no clamp), so argmax is exact.   in [N]  logits [M]  w [M][N]  b [M]
 * ------------------------------------------------------------------------ */
void fc(const data_t *in, acc_t *logits, const wgt_t *w, const bias_t *b,
        int N, int M)
{
TO: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=1001 max=1001
        acc_t acc = b[to];
    TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
            acc += (acc_t)in[ti] * (acc_t)w[to * N + ti];
        }
        logits[to] = acc;
    }
}
