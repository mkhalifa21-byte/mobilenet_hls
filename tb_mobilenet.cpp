/*
 * tb_mobilenet.cpp  --  testbench (C simulation in g++ or Vitis HLS csim)
 *
 * Part 1: unit tests. Each HLS kernel is run on small random inputs and
 *         compared bit-exactly with the golden reference. Shapes include
 *         stride 1, stride 2 and the odd 7x7 size of the last layers.
 * Part 2: full network. mobilenet_top() vs ref_mobilenet() on a whole
 *         224x224 image; all 1001 logits must match exactly.
 *
 * Weights: if weights.bin, biases.bin, shifts.bin and image.bin are all
 * present in the run directory, they are loaded (real trained model).
 * Otherwise deterministic pseudo-random weights are generated. Random weights
 * give the same latency / resources / correctness check, but the predicted
 * class is meaningless.
 *
 * Binary file formats (raw, little-endian, packed as in mobilenet.h):
 *   weights.bin  int8   x W_TOTAL
 *   biases.bin   int32  x B_TOTAL
 *   shifts.bin   int8   x NUM_CONV
 *   image.bin    int8   x 3*224*224   (Q4.4, [channel][row][col])
 *
 * Returns 0 on PASS, 1 on FAIL (Vitis HLS uses this to decide csim result).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mobilenet.h"
#include "mobilenet_ref.h"

/* ---------------- deterministic random numbers (same on every PC) -------- */
static uint32_t rng_state = 12345u;
static uint32_t rng_u32(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}
static float rng_uniform(float lo, float hi)   /* in [lo, hi) */
{
    return lo + (hi - lo) * (float)(rng_u32() >> 8) / 16777216.0f;
}
static int rng_int(int lo, int hi)             /* in [lo, hi] */
{
    return lo + (int)(rng_u32() % (uint32_t)(hi - lo + 1));
}

static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ---------------- compare helpers --------------------------------------- */
static int compare_data(const char *name, const data_t *got, const data_t *exp,
                        int n)
{
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (got[i] != exp[i]) {
            if (errors < 5)
                printf("    mismatch at %d: got %d expected %d\n",
                       i, got[i], exp[i]);
            errors++;
        }
    }
    int zeros = 0, sat = 0;
    for (int i = 0; i < n; i++) { zeros += exp[i] == 0; sat += exp[i] == RELU6_MAX; }
    printf("  %-34s %s  (%d values, %4.1f%% zero, %4.1f%% at 6.0)\n", name,
           errors ? "FAIL" : "PASS", n, 100.0 * zeros / n, 100.0 * sat / n);
    return errors;
}

static int compare_acc(const char *name, const acc_t *got, const acc_t *exp,
                       int n)
{
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (got[i] != exp[i]) {
            if (errors < 5)
                printf("    mismatch at %d: got %d expected %d\n",
                       i, (int)got[i], (int)exp[i]);
            errors++;
        }
    }
    printf("  %-34s %s  (%d values)\n", name, errors ? "FAIL" : "PASS", n);
    return errors;
}

static void fill_data(data_t *a, int n, int lo, int hi)
{ for (int i = 0; i < n; i++) a[i] = (data_t)rng_int(lo, hi); }
static void fill_wgt(wgt_t *a, int n)
{ for (int i = 0; i < n; i++) a[i] = (wgt_t)rng_int(-64, 63); }
static void fill_bias(bias_t *a, int n)
{ for (int i = 0; i < n; i++) a[i] = rng_int(-2000, 2000); }

/* ======================= PART 1: unit tests ============================== */
static int test_conv3x3(int N, int M, int H, int S)
{
    const int Ho = out_size(H, S), shift = 7;
    data_t *in  = (data_t *)malloc(N * H * H);
    wgt_t  *w   = (wgt_t  *)malloc(M * N * 9);
    bias_t *b   = (bias_t *)malloc(M * sizeof(bias_t));
    data_t *got = (data_t *)malloc(M * Ho * Ho);
    data_t *exp = (data_t *)malloc(M * Ho * Ho);
    fill_data(in, N * H * H, -16, 15); fill_wgt(w, M * N * 9); fill_bias(b, M);

    conv3x3(in, got, w, b, N, M, H, S, shift);
    ref_conv2d(in, N, H, w, b, M, 3, S, 1, shift, exp);

    char name[80];
    sprintf(name, "conv3x3   N=%d M=%d H=%d S=%d", N, M, H, S);
    int e = compare_data(name, got, exp, M * Ho * Ho);
    free(in); free(w); free(b); free(got); free(exp);
    return e;
}

static int test_dwconv3x3(int C, int H, int S)
{
    const int Ho = out_size(H, S), shift = 6;
    data_t *in  = (data_t *)malloc(C * H * H);
    wgt_t  *w   = (wgt_t  *)malloc(C * 9);
    bias_t *b   = (bias_t *)malloc(C * sizeof(bias_t));
    data_t *got = (data_t *)malloc(C * Ho * Ho);
    data_t *exp = (data_t *)malloc(C * Ho * Ho);
    fill_data(in, C * H * H, 0, RELU6_MAX); fill_wgt(w, C * 9); fill_bias(b, C);

    dwconv3x3(in, got, w, b, C, H, S, shift);
    ref_conv2d(in, C, H, w, b, C, 3, S, C, shift, exp);

    char name[80];
    sprintf(name, "dwconv3x3 C=%d H=%d S=%d", C, H, S);
    int e = compare_data(name, got, exp, C * Ho * Ho);
    free(in); free(w); free(b); free(got); free(exp);
    return e;
}

static int test_conv1x1(int N, int M, int H)
{
    const int shift = 9;
    data_t *in  = (data_t *)malloc(N * H * H);
    wgt_t  *w   = (wgt_t  *)malloc(M * N);
    bias_t *b   = (bias_t *)malloc(M * sizeof(bias_t));
    data_t *got = (data_t *)malloc(M * H * H);
    data_t *exp = (data_t *)malloc(M * H * H);
    fill_data(in, N * H * H, 0, RELU6_MAX); fill_wgt(w, M * N); fill_bias(b, M);

    conv1x1(in, got, w, b, N, M, H, shift);
    ref_conv2d(in, N, H, w, b, M, 1, 1, 1, shift, exp);

    char name[80];
    sprintf(name, "conv1x1   N=%d M=%d H=%d", N, M, H);
    int e = compare_data(name, got, exp, M * H * H);
    free(in); free(w); free(b); free(got); free(exp);
    return e;
}

static int test_avgpool(int C, int H)
{
    data_t *in = (data_t *)malloc(C * H * H);
    data_t got[1024], exp[1024];
    fill_data(in, C * H * H, 0, RELU6_MAX);
    avgpool(in, got, C, H);
    ref_avgpool(in, C, H, exp);
    char name[80];
    sprintf(name, "avgpool   C=%d H=%d", C, H);
    int e = compare_data(name, got, exp, C);
    free(in);
    return e;
}

static int test_fc(int N, int M)
{
    data_t in[1024]; acc_t got[1024], exp[1024];
    wgt_t  *w = (wgt_t *)malloc(M * N);
    bias_t *b = (bias_t *)malloc(M * sizeof(bias_t));
    fill_data(in, N, 0, RELU6_MAX); fill_wgt(w, M * N); fill_bias(b, M);
    fc(in, got, w, b, N, M);
    ref_fc(in, N, w, b, M, exp);
    char name[80];
    sprintf(name, "fc        N=%d M=%d", N, M);
    int e = compare_acc(name, got, exp, M);
    free(w); free(b);
    return e;
}

/* ======================= PART 2: full network ============================ */

/* Random but "realistic" parameters: weights uniform in +-sqrt(6/fan_in)
 * (He initialization, x1.6 gain, slightly positive bias) so activations neither die nor all saturate, and the
 * per-layer shift is the largest that still fits the weights in int8. */
static void make_random_params(wgt_t *W, bias_t *B, int8_t *shifts)
{
    int cin = IMG_CH, w = 0, b = 0, L = 0;

    /* quantize one layer: count weights, fan_in = inputs per output */
    #define QUANT_LAYER(count, nbias, fan_in)                                  \
    do {                                                                       \
        float lim = 1.6f * sqrtf(6.0f / (float)(fan_in)), mx = 0.0f;                  \
        float *tmp = (float *)malloc((count) * sizeof(float));                 \
        for (int k = 0; k < (count); k++) {                                    \
            tmp[k] = rng_uniform(-lim, lim);                                   \
            if (fabsf(tmp[k]) > mx) mx = fabsf(tmp[k]);                        \
        }                                                                      \
        int sh = (int)floorf(log2f(127.0f / mx));                              \
        sh = clampi(sh, 1, 14);                                                \
        if (L < NUM_CONV) shifts[L] = (int8_t)sh;                              \
        for (int k = 0; k < (count); k++)                                      \
            W[w + k] = (wgt_t)clampi((int)lrintf(tmp[k] * (1 << sh)), -128, 127); \
        for (int k = 0; k < (nbias); k++)                                      \
            B[b + k] = (bias_t)lrintf(rng_uniform(-0.05f, 0.25f) *             \
                                      (float)(1 << (FA + sh)));                \
        free(tmp);                                                             \
        w += (count); b += (nbias); L++;                                       \
    } while (0)

    for (int i = 0; i < NUM_DEFS; i++) {
        const int d = scaled_depth(MOBILENET_DEFS[i].depth);
        if (MOBILENET_DEFS[i].type == LT_CONV) {
            QUANT_LAYER(d * cin * 9, d, cin * 9);
        } else {
            QUANT_LAYER(cin * 9, cin, 9);          /* depthwise */
            QUANT_LAYER(cin * d, d, cin);          /* pointwise */
        }
        cin = d;
    }
    QUANT_LAYER(cin * NUM_CLASSES, NUM_CLASSES, cin);   /* fc */
    #undef QUANT_LAYER
}

static int load_file(const char *path, void *dst, size_t bytes)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t got = fread(dst, 1, bytes, f);
    fclose(f);
    if (got != bytes) {
        printf("  %s: expected %zu bytes, got %zu -- ignoring\n", path, bytes, got);
        return 0;
    }
    return 1;
}

static void trace_layer(int layer, const char *kind, const data_t *out, int n)
{
    int zeros = 0, sat = 0; long sum = 0;
    for (int i = 0; i < n; i++) {
        zeros += out[i] == 0; sat += out[i] == RELU6_MAX; sum += out[i];
    }
    printf("    layer %2d %-4s %7d values  mean %5.2f  %4.1f%% zero  %4.1f%% at 6.0\n",
           layer, kind, n, (double)sum / n / (1 << FA),
           100.0 * zeros / n, 100.0 * sat / n);
}

static void top5(const acc_t *logits, int *idx)
{
    for (int k = 0; k < 5; k++) {
        int best = -1;
        for (int i = 0; i < NUM_CLASSES; i++) {
            int used = 0;
            for (int j = 0; j < k; j++) used |= (idx[j] == i);
            if (!used && (best < 0 || logits[i] > logits[best])) best = i;
        }
        idx[k] = best;
    }
}

int main(void)
{
    int errors = 0;

    /* sanity check of the packed sizes in mobilenet.h */
    int wt, bt;
    ref_param_totals(&wt, &bt);
    printf("Parameter totals: weights %d (W_TOTAL %d), biases %d (B_TOTAL %d)\n",
           wt, W_TOTAL, bt, B_TOTAL);
    if (wt != W_TOTAL || bt != B_TOTAL) {
        printf("ERROR: update W_TOTAL / B_TOTAL in mobilenet.h (and the m_axi "
               "depth= values in mobilenet_top.cpp)\n");
        return 1;
    }

    /* ---------------- Part 1 ---------------- */
    printf("\n=== Part 1: kernel unit tests (bit-exact vs reference) ===\n");
    errors += test_conv3x3(3, 16, 16, 2);   /* like Conv2d_0, small        */
    errors += test_conv3x3(3, 8, 9, 1);     /* odd size, stride 1          */
    errors += test_dwconv3x3(16, 8, 1);
    errors += test_dwconv3x3(16, 8, 2);     /* asymmetric SAME padding     */
    errors += test_dwconv3x3(32, 7, 1);     /* 7x7 like the last layers    */
    errors += test_dwconv3x3(8, 14, 2);
    errors += test_conv1x1(16, 32, 8);
    errors += test_conv1x1(64, 16, 7);
    errors += test_avgpool(32, 7);
    errors += test_fc(64, 10);

    /* ---------------- Part 2 ---------------- */
    printf("\n=== Part 2: full MobileNetV1 (alpha=%d/%d, %dx%d) ===\n",
           ALPHA_NUM, ALPHA_DEN, IMG_SIZE, IMG_SIZE);

    const int IMG_N = IMG_CH * IMG_SIZE * IMG_SIZE;
    data_t *image   = (data_t *)malloc(IMG_N);
    wgt_t  *weights = (wgt_t  *)malloc(W_TOTAL);
    bias_t *biases  = (bias_t *)malloc(B_TOTAL * sizeof(bias_t));
    int8_t  shifts[NUM_CONV];
    data_t *fm      = (data_t *)calloc(FM_TOTAL, 1);
    data_t *ref_a   = (data_t *)malloc(FM_MAX);
    data_t *ref_b   = (data_t *)malloc(FM_MAX);
    acc_t   logits_hw[NUM_CLASSES], logits_ref[NUM_CLASSES];

    int real = load_file("weights.bin", weights, W_TOTAL) &&
               load_file("biases.bin",  biases,  B_TOTAL * sizeof(bias_t)) &&
               load_file("shifts.bin",  shifts,  NUM_CONV) &&
               load_file("image.bin",   image,   IMG_N);
    if (real) {
        printf("  Loaded trained weights + image from .bin files\n");
    } else {
        printf("  No .bin files found -> using random weights and image\n");
        make_random_params(weights, biases, shifts);
        for (int i = 0; i < IMG_N; i++)               /* image in [-1, 1) */
            image[i] = (data_t)lrintf(rng_uniform(-1.0f, 1.0f) * (1 << FA));
    }

    printf("  Reference model, per-layer output statistics:\n");
    clock_t t0 = clock();
    ref_mobilenet(image, weights, biases, shifts, logits_ref, ref_a, ref_b,
                  trace_layer);
    clock_t t1 = clock();
    mobilenet_top(image, fm, weights, biases, shifts, logits_hw);
    clock_t t2 = clock();
    printf("  CPU time: reference %.2f s, HLS C model %.2f s\n",
           (double)(t1 - t0) / CLOCKS_PER_SEC, (double)(t2 - t1) / CLOCKS_PER_SEC);

    errors += compare_acc("mobilenet_top logits", logits_hw, logits_ref,
                          NUM_CLASSES);

    int t5_hw[5], t5_ref[5];
    top5(logits_hw, t5_hw); top5(logits_ref, t5_ref);
    printf("  Top-5 classes  HLS: %d %d %d %d %d   reference: %d %d %d %d %d\n",
           t5_hw[0], t5_hw[1], t5_hw[2], t5_hw[3], t5_hw[4],
           t5_ref[0], t5_ref[1], t5_ref[2], t5_ref[3], t5_ref[4]);
    if (!real) printf("  (random weights: class numbers are meaningless)\n");

    free(image); free(weights); free(biases); free(fm); free(ref_a); free(ref_b);

    printf("\n%s  (%d mismatching values)\n",
           errors ? "*** TEST FAILED ***" : "*** ALL TESTS PASSED ***", errors);
    return errors ? 1 : 0;
}
