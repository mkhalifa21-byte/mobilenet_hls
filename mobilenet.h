/*
 * mobilenet.h  --  MobileNetV1 (alpha = 0.5, 160x160) for Vitis HLS
 * EECS 298 Deep Learning Accelerator Design -- final project
 *
 * Translated by hand from TF-slim mobilenet_v1.py (MOBILENETV1_CONV_DEFS,
 * mobilenet_v1_base(), mobilenet_v1()).
 *
 * ---------------------------------------------------------------------------
 * NUMBER FORMAT (all integer, so it is exact and identical in C sim, RTL and
 * on the board)
 *
 *   data_t   int8   feature maps,  FA = 4 fractional bits (Q4.4)
 *                   real value = stored / 16   ->  range -8.0 .. +7.9375
 *                   (covers the [-1,1] input image and the [0,6] ReLU6 range)
 *   wgt_t    int8   weights, per-layer fractional bits = shift[layer]
 *                   real value = stored / 2^shift
 *   bias_t   int32  bias, stored at the accumulator scale (FA + shift bits)
 *   acc_t    int32  accumulator: sum of data*weight has FA + shift frac bits
 *
 *   Output of a layer:  out = clamp( (acc + 2^(shift-1)) >> shift )
 *     - adding 2^(shift-1) before shifting = round to nearest
 *     - ">> shift" removes the weight's fractional bits, so the result is
 *       back in Q4.4
 *     - clamp to [0, 6.0] = ReLU6   (6.0 in Q4.4 = 6 * 16 = 96)
 *
 * MEMORY LAYOUT (matches the Lecture 2 loop nests)
 *   feature map         [channel][row][col]
 *   standard conv w     [out_ch][in_ch][3][3]
 *   depthwise w         [ch][3][3]
 *   pointwise / fc w    [out_ch][in_ch]
 *
 * All weights of all layers are packed back-to-back in ONE array, in network
 * order: conv0, dw1, pw1, dw2, pw2, ..., dw13, pw13, fc. Same for biases.
 * shifts[] has one entry per conv layer (27 entries; fc reuses none).
 * ---------------------------------------------------------------------------
 */
#ifndef MOBILENET_H
#define MOBILENET_H

#include <stdint.h>

typedef int16_t  data_t;
typedef int8_t  wgt_t;
typedef int32_t bias_t;
typedef int32_t acc_t;

#define FA         8                 /* fractional bits of data_t        */
#define RELU6_MAX  (6 << FA)         /* 6.0 in Q4.4 = 96                 */

/* ---- width multiplier alpha = ALPHA_NUM / ALPHA_DEN (0.5) ------------- */
#define ALPHA_NUM  1
#define ALPHA_DEN  2
#define MIN_DEPTH  8

/* ---- network constants ------------------------------------------------ */
#define IMG_SIZE     160
#define IMG_CH       3
#define NUM_CLASSES  1001            /* slim checkpoints: class 0 = background */
#define NUM_DEFS     14              /* entries in MOBILENET_DEFS          */
#define NUM_CONV     27              /* conv0 + 13 x (dw + pw)             */
#define S_TOTAL      5472            /* one shift per conv output channel = B_TOTAL - NUM_CLASSES */

/* Largest feature map = output of layer 1 (Conv2d_1_pointwise):
 *   (IMG_SIZE/2) x (IMG_SIZE/2) x (64*alpha)
 *   160 input: 80 x 80 x 32 = 204,800     (224 input: 112 x 112 x 32 = 401,408) */
#define FM_MAX   ((IMG_SIZE / 2) * (IMG_SIZE / 2) * (64 * ALPHA_NUM / ALPHA_DEN))

/* The kernel uses one DDR buffer "fm" split into three regions. */
#define REGION_A    0
#define REGION_B    (FM_MAX)
#define REGION_TMP  (2 * FM_MAX)
#define FM_TOTAL    (3 * FM_MAX)

/* m_axi depth= values in mobilenet_top.cpp must equal these (HLS pragmas
 * need plain numbers; the testbench checks they still match):
 *   image depth = IMG_CH * IMG_SIZE * IMG_SIZE = 3 * 160 * 160 = 76,800
 *   fm depth    = FM_TOTAL = 3 * 204,800                       = 614,400 */
#define IMG_DEPTH_PRAGMA  76800
#define FM_DEPTH_PRAGMA   614400

/* Totals for alpha = 0.5 (checked by the testbench at start-up).
 * If you change alpha, the testbench prints the new values to put here. */
#define W_TOTAL  1320160             /* 807,648 conv + 512,512 fc          */
#define B_TOTAL  6473

/* ---- layer table: direct copy of MOBILENETV1_CONV_DEFS (lines 123-138) -- */
enum { LT_CONV = 0, LT_DWSEP = 1 };

typedef struct {
    int type;     /* LT_CONV or LT_DWSEP                        */
    int stride;   /* 1 or 2                                     */
    int depth;    /* output channels at alpha = 1.0             */
} layer_def_t;

static const layer_def_t MOBILENET_DEFS[NUM_DEFS] = {
    {LT_CONV,  2,   32},
    {LT_DWSEP, 1,   64},
    {LT_DWSEP, 2,  128},
    {LT_DWSEP, 1,  128},
    {LT_DWSEP, 2,  256},
    {LT_DWSEP, 1,  256},
    {LT_DWSEP, 2,  512},
    {LT_DWSEP, 1,  512},
    {LT_DWSEP, 1,  512},
    {LT_DWSEP, 1,  512},
    {LT_DWSEP, 1,  512},
    {LT_DWSEP, 1,  512},
    {LT_DWSEP, 2, 1024},
    {LT_DWSEP, 1, 1024},
};

/* depth = max(int(d * alpha), min_depth)   (mobilenet_v1.py line 217) */
static inline int scaled_depth(int d)
{
    int x = d * ALPHA_NUM / ALPHA_DEN;
    return x > MIN_DEPTH ? x : MIN_DEPTH;
}

/* TF 'SAME' padding for a 3x3 kernel.
 *   H_out     = ceil(H / S)
 *   pad_total = max((H_out - 1) * S + K - H, 0)
 *   pad_begin = pad_total / 2      (top / left; the rest goes bottom / right)
 * Example: H = 160, S = 2 -> H_out = 80, pad_total = 1, pad_begin = 0.
 *          H = 80,  S = 1 -> H_out = 80, pad_total = 2, pad_begin = 1.   */
static inline int out_size(int H, int S) { return (H + S - 1) / S; }

static inline int same_pad_begin(int H, int S, int K)
{
    int Ho = out_size(H, S);
    int t  = (Ho - 1) * S + K - H;
    if (t < 0) t = 0;
    return t / 2;
}

/* Round, remove weight fractional bits, clamp (ReLU6 or plain int8). */
static inline data_t requant(acc_t acc, int shift, int relu6)
{
    acc_t rnd = (shift > 0) ? ((acc_t)1 << (shift - 1)) : 0;
    acc_t r   = (acc + rnd) >> shift;
    acc_t lo  = relu6 ? 0 : -32768;
    acc_t hi  = relu6 ? RELU6_MAX : 32767;        // come back to this if theres an issue
    if (r < lo) r = lo;
    if (r > hi) r = hi;
    return (data_t)r;
}

/* ---- HLS kernels (mobilenet_kernels.cpp) ------------------------------ */
void conv3x3(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
             int N, int M, int H, int S, const int8_t *shift);
void dwconv3x3(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
               int C, int H, int S, const int8_t *shift);
void conv1x1(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
             int N, int M, int H, const int8_t *shift);
void avgpool(const data_t *in, data_t *out, int C, int H);
void fc(const data_t *in, acc_t *logits, const wgt_t *w, const bias_t *b,
        int N, int M);

/* ---- HLS top function (mobilenet_top.cpp) ----------------------------- */
void mobilenet_top(const data_t *image,     /* [3][IMG_SIZE][IMG_SIZE] */
                   data_t       *fm,        /* [FM_TOTAL] scratch DDR */
                   const wgt_t  *weights,   /* [W_TOTAL]              */
                   const bias_t *biases,    /* [B_TOTAL]              */
                   const int8_t *shifts,    /* [NUM_CONV]             */
                   acc_t        *logits);   /* [NUM_CLASSES]          */

#endif
