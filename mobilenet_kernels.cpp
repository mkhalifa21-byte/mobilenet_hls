/*
 * mobilenet_kernels.cpp  --  HLS kernels
 *
 * Status of each kernel:
 *   conv3x3   OPTIMIZED  row band buffer, 4 outputs x 27 MACs/cycle (v3)
 *   dwconv3x3 OPTIMIZED  on-chip plane + 3x3 window, 9 MACs / cycle (v1)
 *   v3: conv3x3 does 4 output channels per pass instead of 16 at once
 *       (v1: 433 DSPs; an ALLOCATION limit was ignored). dwconv3x3 is the
 *       v1 code (v2/v2.1 address rewrites crashed Vitis HLS 2023.1).
 *   conv1x1   baseline   (local buffers / tiling: teammate's work)
 *   avgpool   baseline   (0.01% of runtime, not worth optimizing)
 *   fc        baseline   (limited by reading 1 weight byte per cycle from
 *                         DDR; speeds up with the 128-bit m_axi step)
 *
 * Every kernel keeps the same function signature as the baseline and must
 * still pass the testbench bit-exactly (make run -> ALL TESTS PASSED).
 *
 * LOOP_TRIPCOUNT only tells the HLS report how many iterations to assume for
 * variable-bound loops (it does not change the hardware).
 */
#include "mobilenet.h"

/* ------------------------------------------------------------------------
 * Standard 3x3 convolution (only Conv2d_0: 3 -> 16 channels, 160 -> 80)
 *   in  [N][H][H]   out [M][Ho][Ho]   w [M][N][3][3]   b [M]
 *
 * OPTIMIZATION (baseline: 1 MAC per cycle, 2.76 M cycles for Conv2d_0)
 * For each output row:
 *   1. LOAD_BAND : copy the 3 input rows it needs (x all N channels) from DDR
 *                  into an on-chip band with a zero column on each side.
 *                  9 banks, one per (channel, row).
 *   2. COMPUTE   : C0_TG = 4 output channels at a time (4 passes for 16).
 *                  Each pass streams across the band one column per cycle
 *                  (9 reads, one per bank), slides a 3 x 3 x N window in
 *                  registers, and when the window sits on an output position
 *                  computes 4 outputs x 27 MACs = 108 MACs (~108 DSPs).
 *   3. WRITE_ROW : write the finished row, channel by channel (bursts).
 * v1 computed all 16 channels at once (II=4); HLS built 432 multipliers
 * (433 DSPs) and an ALLOCATION limit was ignored because the multiply-adds
 * become fused DSP multiply-accumulate units. Fewer channels per pass is
 * the structural way to use fewer DSPs.
 * Expected per output row: 9*(H+2) + (M/4)*(H+2 + ~12) + M*Ho cycles
 *   Conv2d_0: (1458 + 4*174 + 1280) x 80 rows = ~275 K cycles.
 * Limits: N <= C0_MAXN, M <= C0_MAXM, H <= C0_MAXW.
 * ------------------------------------------------------------------------ */
#define C0_MAXN 3
#define C0_MAXM 16
#define C0_MAXW 160
#define C0_TG   4                       /* output channels per pass */
#define C0_NG   (C0_MAXM / C0_TG)       /* max passes                */

void conv3x3(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
             int N, int M, int H, int S, const int8_t *shift)
{
    const int Ho  = out_size(H, S);
    const int pad = same_pad_begin(H, S, 3);
    const int NG  = (M + C0_TG - 1) / C0_TG;      /* passes this layer   */
    const int NQ  = S * (Ho - 1) - pad + 4;       /* band columns walked */

    /* weights / bias / shift, indexed [k][g] = output channel g*C0_TG + k */
    wgt_t  wl[C0_TG][C0_NG][C0_MAXN * 9];
    bias_t bl[C0_TG][C0_NG];
    int8_t sl[C0_TG][C0_NG];
#pragma HLS ARRAY_PARTITION variable=wl type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=wl type=complete dim=3
#pragma HLS ARRAY_PARTITION variable=bl type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=sl type=complete dim=1

    /* band[ci][r][x+1] = input row (S*oy + r - pad), column x; columns 0 and
     * H+1 are the zero padding. One bank per (ci, r). */
    data_t band[C0_MAXN][3][C0_MAXW + 2];
#pragma HLS ARRAY_PARTITION variable=band type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=band type=complete dim=2

    /* finished output row: orow[k][g][ox] = channel g*C0_TG + k */
    data_t orow[C0_TG][C0_NG][C0_MAXW];
#pragma HLS ARRAY_PARTITION variable=orow type=complete dim=1

LOAD_W: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        const int k = to % C0_TG, g = to / C0_TG;
        bl[k][g] = b[to];
        sl[k][g] = shift[to];
    LOAD_WK: for (int t = 0; t < N * 9; t++) {
#pragma HLS LOOP_TRIPCOUNT min=27 max=27
#pragma HLS PIPELINE II=1
            wl[k][g][t] = w[to * N * 9 + t];
        }
    }

ROWS: for (int oy = 0; oy < Ho; oy++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80

    LOAD_BAND: for (int ci = 0; ci < N; ci++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
        LOAD_R: for (int r = 0; r < 3; r++) {
            const int  y      = S * oy + r - pad;
            const bool row_ok = (y >= 0 && y < H);
        LOAD_X: for (int x = 0; x < H + 2; x++) {
#pragma HLS LOOP_TRIPCOUNT min=162 max=162
#pragma HLS PIPELINE II=1
                data_t v = 0;
                if (row_ok && x >= 1 && x <= H)
                    v = in[(ci * H + y) * H + (x - 1)];
                band[ci][r][x] = v;
            }
        }
    }

    PASS: for (int g = 0; g < NG; g++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4
        /* window over band columns q-2 .. q, all channels and rows */
        data_t win[C0_MAXN][3][3];
#pragma HLS ARRAY_PARTITION variable=win type=complete dim=0

        COMPUTE: for (int q = 0; q < NQ; q++) {
#pragma HLS LOOP_TRIPCOUNT min=162 max=162
#pragma HLS PIPELINE II=1
            /* bring in band column q (one read per bank) */
            for (int ci = 0; ci < C0_MAXN; ci++)
                for (int r = 0; r < 3; r++) {
                    win[ci][r][0] = win[ci][r][1];
                    win[ci][r][1] = win[ci][r][2];
                    win[ci][r][2] = (ci < N) ? band[ci][r][q] : (data_t)0;
                }

            /* window left column = band col q-2 = input x0 + 1, and an
             * output ox needs x0 = S*ox - pad  ->  q = S*ox - pad + 3 */
            const int t  = q + pad - 3;
            const int ox = (S == 1) ? t : (t >> 1);
            const bool hit = (t >= 0) && (S == 1 || (t & 1) == 0) && (ox < Ho);
            if (hit) {
                for (int k = 0; k < C0_TG; k++) {
                    acc_t acc = bl[k][g];
                    for (int ci = 0; ci < C0_MAXN; ci++)
                        for (int r = 0; r < 3; r++)
                            for (int kj = 0; kj < 3; kj++)
                                acc += (acc_t)win[ci][r][kj] *
                                       (acc_t)wl[k][g][(ci * 3 + r) * 3 + kj];
                    if (g * C0_TG + k < M)
                        orow[k][g][ox] = requant(acc, sl[k][g], 1);
                }
            }
        }
    }

    WRITE_ROW: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        const int k = to % C0_TG, g = to / C0_TG;
        WRITE_X: for (int ox = 0; ox < Ho; ox++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
#pragma HLS PIPELINE II=1
            out[(to * Ho + oy) * Ho + ox] = orow[k][g][ox];
        }
    }
}
}

/* ------------------------------------------------------------------------
 * Depthwise 3x3 convolution: output channel c only looks at input channel c
 * (slim.separable_conv2d with num_outputs=None).
 *   in [C][H][H]   out [C][Ho][Ho]   w [C][3][3]   b [C]
 *
 * OPTIMIZATION (baseline: 27 cycles per output, 13.3 M cycles total)
 * One channel at a time, in two separate loops:
 *   1. FETCH   : burst-read the whole H x H channel plane from DDR into an
 *                on-chip buffer (1 pixel / cycle, sequential addresses).
 *                The buffer has a 1-pixel zero border, so the compute loop
 *                never checks image edges (TF 'SAME' padding for free).
 *   2. COMPUTE : slide a 3x3 window over the plane. Each cycle reads one new
 *                column (3 pixels, one from each of the 3 row banks) and does
 *                all 9 MACs (9 DSPs), PIPELINE II=1.
 * Why two loops: the compute loop only touches on-chip memory, so its
 * pipeline is short and has no DDR latency inside it. (A single loop with a
 * line buffer fed straight from DDR passes C simulation but can read stale
 * line-buffer data in hardware on the small 5x5 layers.)
 * Iteration (r, q) of COMPUTE puts input row r-pad, column q-pad in the
 * window's bottom-right corner; it is an output when (r-2) and (q-2) are
 * divisible by the stride: oy = (r-2)/S, ox = (q-2)/S.
 * Expected cycles per channel: H*H (fetch) + (NI-2)*NI (compute) + ~100,
 *   NI = S*(Ho-1)+3.  Whole network: ~1.9 M cycles (~7x faster).
 * When feature maps move on chip, FETCH can be removed and COMPUTE reads the
 * on-chip feature map directly.
 * Limit: H <= DW_MAXH.
 * ------------------------------------------------------------------------ */
#define DW_MAXH 80

void dwconv3x3(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
               int C, int H, int S, const int8_t *shift)
{
    const int Ho  = out_size(H, S);
    const int pad = same_pad_begin(H, S, 3);
    const int NI  = S * (Ho - 1) + 3;     /* window positions per row/col */

    /* plane[y+1][x+1] = input pixel (y, x); row/col 0 and H+1 are zero.
     * 3 row banks (cyclic): rows pr-2, pr-1, pr are always in different
     * banks, so one column of the window is read in a single cycle. */
    data_t plane[DW_MAXH + 2][DW_MAXH + 2];
#pragma HLS ARRAY_PARTITION variable=plane type=cyclic factor=3 dim=1

    /* zero the border for this layer's size (FETCH only writes the inside,
     * and a previous, larger layer may have left data where the border is) */
ZERO_TB: for (int i = 0; i < H + 2; i++) {
#pragma HLS LOOP_TRIPCOUNT min=7 max=82
#pragma HLS PIPELINE II=1
        plane[0][i]     = 0;
        plane[H + 1][i] = 0;
    }
ZERO_LR: for (int i = 0; i < H + 2; i++) {
#pragma HLS LOOP_TRIPCOUNT min=7 max=82
#pragma HLS PIPELINE II=1
        plane[i][0]     = 0;
        plane[i][H + 1] = 0;
    }

CH: for (int c = 0; c < C; c++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512
        wgt_t wr[9];
#pragma HLS ARRAY_PARTITION variable=wr type=complete dim=1
    LOAD_W: for (int k = 0; k < 9; k++) {
#pragma HLS PIPELINE II=1
            wr[k] = w[c * 9 + k];
        }
        const acc_t bc = b[c];
        const int   sh = shift[c];

        /* 1. FETCH: sequential DDR addresses -> burst reads (v1 code, which
         * synthesizes; v2/v2.1 address rewrites crashed Vitis HLS 2023.1) */
    FETCH_Y: for (int y = 0; y < H; y++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
        FETCH_X: for (int x = 0; x < H; x++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
#pragma HLS PIPELINE II=1
                plane[y + 1][x + 1] = in[(c * H + y) * H + x];
            }
        }

        /* 2. COMPUTE: 3x3 window in registers, 9 MACs per cycle */
        data_t win[3][3];
#pragma HLS ARRAY_PARTITION variable=win type=complete dim=0
    ROW: for (int r = 2; r < NI; r++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
        COL: for (int q = 0; q < NI; q++) {
#pragma HLS LOOP_TRIPCOUNT min=7 max=82
#pragma HLS PIPELINE II=1
            const int pr = r + 1 - pad;     /* plane row of window bottom  */
            const int pc = q + 1 - pad;     /* plane col of the new column */

            /* slide the window one column left, bring in column pc */
            for (int i = 0; i < 3; i++) {
                win[i][0] = win[i][1];
                win[i][1] = win[i][2];
            }
            win[0][2] = plane[pr - 2][pc];
            win[1][2] = plane[pr - 1][pc];
            win[2][2] = plane[pr][pc];

            /* is this window an output position? */
            const int ty = r - 2, tx = q - 2;
            const bool ok_y = (S == 1) || ((ty & 1) == 0);
            const bool ok_x = (tx >= 0) && (S == 1 || (tx & 1) == 0);
            const int oy = (S == 1) ? ty : (ty >> 1);
            const int ox = (S == 1) ? tx : (tx >> 1);
            if (ok_y && ok_x && oy < Ho && ox < Ho) {
                acc_t acc = bc;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        acc += (acc_t)win[i][j] * (acc_t)wr[i * 3 + j];
                out[(c * Ho + oy) * Ho + ox] = requant(acc, sh, 1);
            }
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
