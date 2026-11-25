/*
 * mobilenet_kernels.cpp  --  HLS kernels
 *
 * Status of each kernel:
 *   conv3x3   OPTIMIZED  row band buffer, 4 outputs x 27 MACs/cycle (v3)
 *   dwconv3x3 OPTIMIZED  on-chip plane + 3x3 window, 9 MACs / cycle (v1)
 *   conv1x1   OPTIMIZED  URAM input buffer, 16x16 MAC array          (v4)
 *   avgpool   baseline   (0.01% of runtime, not worth optimizing)
 *   fc        OPTIMIZED  128-bit weights, 16 MACs / cycle             (v5)
 *
 * History:
 *   v1: conv3x3 (16 outputs at II=4, 433 DSPs) + dwconv3x3 optimized.
 *   v2/v2.1: dwconv3x3 address rewrites crashed Vitis HLS 2023.1;
 *       an ALLOCATION limit on conv3x3 was ignored. Reverted.
 *   v3: conv3x3 does 4 output channels per pass -> 109 DSPs.
 *   v4: conv1x1 optimized (16 output x 16 input channels per cycle,
 *       input feature map in URAM, weights per output tile on chip).
 *   v5: 128-bit weight port: every kernel reads 16 weights per cycle
 *       (wvec_t); fc does 16 MACs per cycle from an on-chip input copy.
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

void conv3x3(const data_t *in, data_t *out, const wvec_t *w, const bias_t *b,
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

    /* v5: read all M*N*9 weights 16 per cycle (128-bit words) into a
     * staging buffer, then spread them into the register file */
    wgt_t stage[C0_MAXM * C0_MAXN * 9 + WVEC];
#pragma HLS ARRAY_RESHAPE variable=stage type=cyclic factor=16 dim=1
    const int nw = (M * N * 9 + WVEC - 1) / WVEC;
LOAD_WV: for (int i = 0; i < nw; i++) {
#pragma HLS LOOP_TRIPCOUNT min=27 max=27
#pragma HLS PIPELINE II=1
        const wvec_t word = w[i];
        for (int j = 0; j < WVEC; j++)
            stage[i * WVEC + j] = word.v[j];
    }

LOAD_W: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        const int k = to % C0_TG, g = to / C0_TG;
        bl[k][g] = b[to];
        sl[k][g] = shift[to];
    LOAD_WK: for (int t = 0; t < N * 9; t++) {
#pragma HLS LOOP_TRIPCOUNT min=27 max=27
#pragma HLS PIPELINE II=1
            wl[k][g][t] = stage[to * N * 9 + t];
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
#define DW_MAXC 512

void dwconv3x3(const data_t *in, data_t *out, const wvec_t *w, const bias_t *b,
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

    /* v5: all C*9 weights of the layer, read 16 per cycle (128-bit) */
    wgt_t wall[DW_MAXC * 9 + WVEC];
#pragma HLS ARRAY_RESHAPE variable=wall type=cyclic factor=16 dim=1
    const int nw = (C * 9 + WVEC - 1) / WVEC;
LOAD_WV: for (int i = 0; i < nw; i++) {
#pragma HLS LOOP_TRIPCOUNT min=9 max=288
#pragma HLS PIPELINE II=1
        const wvec_t word = w[i];
        for (int j = 0; j < WVEC; j++)
            wall[i * WVEC + j] = word.v[j];
    }

CH: for (int c = 0; c < C; c++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512
        wgt_t wr[9];
#pragma HLS ARRAY_PARTITION variable=wr type=complete dim=1
    LOAD_W: for (int k = 0; k < 9; k++) {
#pragma HLS PIPELINE II=1
            wr[k] = wall[c * 9 + k];
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
 *
 * OPTIMIZATION (baseline: N+33 cycles per output value, 91.6 M cycles,
 * 85% of the whole network)
 *   1. LOAD_IN  : burst-read the whole input feature map (N x H*H, at most
 *                 102,400 values) once into an on-chip buffer in URAM,
 *                 split into PW_TN = 16 banks (bank = input channel % 16).
 *   2. For each tile of PW_TM = 16 output channels:
 *      a. LOAD_W  : read that tile's 16 x N weights (sequential DDR) into
 *                   registers; 16 weights of a row are packed in one word.
 *      b. COMPUTE : per pixel, walk the input channels 16 at a time:
 *                   16 outputs x 16 inputs = 256 MACs per cycle, II=1
 *                   (~256 DSPs). Partial sums stay in registers.
 *      c. WRITE   : finished pixels (PW_TP at a time) go to a small output
 *                   buffer and are burst-written to DDR channel by channel.
 * Each input value is read from DDR once (not once per output channel),
 * each weight once, each output written once.
 * Expected cycles per layer ~ N*HW (load) + (M/16) * (16*N + HW*N/16
 *   + 16*HW + fills)   -> about 2.5 M cycles for all 13 layers (~35x).
 *   Most of that is DDR transfers at one value per cycle; the 128-bit
 *   m_axi step shrinks it further.
 * Limits: N <= PW_MAXN, N*H*H <= PW_TN*PW_IBD, N a multiple of 16
 * (v5: each weight row is read as N/16 128-bit words), M any.
 * ------------------------------------------------------------------------ */
#define PW_TM   16          /* output channels in parallel         */
#define PW_TN   16          /* input channels in parallel          */
#define PW_MAXN 512         /* largest input channel count         */
#define PW_IBD  6400        /* input buffer depth per bank: 16 x 6400 = 102,400 = largest N*H*H  */
#define PW_TP   256         /* pixels per output-buffer tile       */

void conv1x1(const data_t *in, data_t *out, const wvec_t *w, const bias_t *b,
             int N, int M, int H, const int8_t *shift)
{
    const int HW = H * H;
    const int NT = (N + PW_TN - 1) / PW_TN;     /* input-channel groups  */
    const int MT = (M + PW_TM - 1) / PW_TM;     /* output-channel tiles  */

    /* ibuf[j][g*HW + p] = input channel (g*16 + j), pixel p */
    data_t ibuf[PW_TN][PW_IBD];
#pragma HLS ARRAY_PARTITION variable=ibuf type=complete dim=1
#pragma HLS BIND_STORAGE variable=ibuf type=ram_2p impl=uram

    /* wt[k][ti]: weight of output (tile*16 + k), input ti. The 16 weights
     * read together (ti = g*16 .. g*16+15) are packed into one word. */
    wgt_t wt[PW_TM][PW_MAXN];
#pragma HLS ARRAY_PARTITION variable=wt type=complete dim=1
#pragma HLS ARRAY_RESHAPE variable=wt type=cyclic factor=16 dim=2

    bias_t bt[PW_TM];
    int8_t st[PW_TM];
    acc_t  acc[PW_TM];
    data_t obuf[PW_TM][PW_TP];
#pragma HLS ARRAY_PARTITION variable=bt   type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=st   type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc  type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=obuf type=complete dim=1

    /* 1. input feature map -> on-chip (sequential DDR addresses) */
LOAD_IN: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512
        const int src  = ti * HW;
        const int dst  = (ti / PW_TN) * HW;
        const int bank = ti % PW_TN;
    LOAD_IN_P: for (int p = 0; p < HW; p++) {
#pragma HLS LOOP_TRIPCOUNT min=25 max=6400
#pragma HLS PIPELINE II=1
            ibuf[bank][dst + p] = in[src + p];
        }
    }

TILE: for (int mt = 0; mt < MT; mt++) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=32
        const int to0 = mt * PW_TM;

        /* 2a. this tile's weights, bias, shift (zero for missing rows/cols) */
    LOAD_WK: for (int k = 0; k < PW_TM; k++) {
            const bool row_ok = (to0 + k < M);
            bt[k] = row_ok ? b[to0 + k] : 0;
            st[k] = row_ok ? shift[to0 + k] : (int8_t)1;
            /* v5: one 128-bit word = 16 weights of this row per cycle */
            const int wbase = (to0 + k) * NT;     /* = (to0+k)*N/16 */
        LOAD_WT: for (int g = 0; g < NT; g++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=32
#pragma HLS PIPELINE II=1
                wvec_t word;
                if (row_ok) word = w[wbase + g];
                for (int j = 0; j < PW_TN; j++)
                    wt[k][g * PW_TN + j] = row_ok ? word.v[j] : (wgt_t)0;
            }
        }

    PTILE: for (int p0 = 0; p0 < HW; p0 += PW_TP) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=25
            const int np = (HW - p0 < PW_TP) ? (HW - p0) : PW_TP;

            /* 2b. 256 MACs per cycle */
        PIX: for (int pp = 0; pp < np; pp++) {
#pragma HLS LOOP_TRIPCOUNT min=25 max=256
            GRP: for (int g = 0; g < NT; g++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=32
#pragma HLS PIPELINE II=1
                    const int addr = g * HW + p0 + pp;
                    data_t x[PW_TN];
#pragma HLS ARRAY_PARTITION variable=x type=complete dim=1
                    for (int j = 0; j < PW_TN; j++)
                        x[j] = ibuf[j][addr];

                    for (int k = 0; k < PW_TM; k++) {
                        acc_t part = 0;
                        for (int j = 0; j < PW_TN; j++)
                            part += (acc_t)wt[k][g * PW_TN + j] * (acc_t)x[j];
                        const acc_t s = ((g == 0) ? bt[k] : acc[k]) + part;
                        acc[k] = s;
                        if (g == NT - 1)
                            obuf[k][pp] = requant(s, st[k], 1);
                    }
                }
            }

            /* 2c. burst-write this pixel tile, one output channel at a time */
        WRITE_K: for (int k = 0; k < PW_TM; k++) {
                if (to0 + k < M) {
                    const int obase = (to0 + k) * HW + p0;
                WRITE_P: for (int pp = 0; pp < np; pp++) {
#pragma HLS LOOP_TRIPCOUNT min=25 max=256
#pragma HLS PIPELINE II=1
                        out[obase + pp] = obuf[k][pp];
                    }
                }
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
 * Kept separate because it outputs raw int32 scores (no ReLU6, no clamp),
 * so argmax is exact.   in [N]  logits [M]  w [M][N]  b [M]
 *
 * OPTIMIZATION (v5; baseline: 1 weight byte per cycle, 530,530 cycles)
 *   - the N pooled inputs are copied on chip once (16 banks)
 *   - each 128-bit weight word gives 16 weights -> 16 MACs per cycle
 *   Expected: 512 + 1001 x (N/16 + ~12) = ~45 K cycles (~12x).
 * Limits: N <= FC_MAXN, N a multiple of 16.
 * ------------------------------------------------------------------------ */
#define FC_MAXN 512

void fc(const data_t *in, acc_t *logits, const wvec_t *w, const bias_t *b,
        int N, int M)
{
    const int NG = N / WVEC;
    data_t x[FC_MAXN];
#pragma HLS ARRAY_PARTITION variable=x type=cyclic factor=16 dim=1

LOAD_X: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
#pragma HLS PIPELINE II=1
        x[ti] = in[ti];
    }

TO: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=1001 max=1001
        acc_t acc = b[to];
    TI: for (int g = 0; g < NG; g++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32
#pragma HLS PIPELINE II=1
            const wvec_t word = w[to * NG + g];
            acc_t part = 0;
            for (int j = 0; j < WVEC; j++)
                part += (acc_t)word.v[j] * (acc_t)x[g * WVEC + j];
            acc += part;
        }
        logits[to] = acc;
    }
}
