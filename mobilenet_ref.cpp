/*
 * mobilenet_ref.cpp  --  GOLDEN REFERENCE (testbench only, never synthesized)
 *
 * This replaces Python as the "answer key". It is written independently of
 * the HLS kernels on purpose: ONE generic grouped convolution handles all
 * three conv types, so a bug in a specialized kernel cannot be copied here.
 *
 *   standard conv : groups = 1,  K = 3
 *   depthwise     : groups = C,  K = 3   (each group has 1 input channel)
 *   pointwise     : groups = 1,  K = 1
 *
 * It uses exactly the same integer arithmetic (same rounding, same clamp),
 * so the HLS kernels must match it BIT-EXACTLY, not just "approximately".
 */
#include "mobilenet_ref.h"

/* Generic grouped 2D conv with TF 'SAME' padding.
 *   in  [N][H][H]
 *   w   [M][N/groups][K][K]
 *   out [M][Ho][Ho]                                                        */
void ref_conv2d(const data_t *in, int N, int H,
                const wgt_t *w, const bias_t *b, int M, int K, int S,
                int groups, const int8_t *shift, data_t *out)
{
    const int Ho  = out_size(H, S);
    const int pad = same_pad_begin(H, S, K);
    const int Ng  = N / groups;      /* input channels per group  */
    const int Mg  = M / groups;      /* output channels per group */

    for (int m = 0; m < M; m++) {
        const int g = m / Mg;        /* which group this output belongs to */
        for (int oy = 0; oy < Ho; oy++) {
            for (int ox = 0; ox < Ho; ox++) {
                acc_t acc = b[m];
                for (int n = 0; n < Ng; n++) {
                    const int ch = g * Ng + n;   /* actual input channel */
                    for (int ky = 0; ky < K; ky++) {
                        const int y = oy * S + ky - pad;
                        if (y < 0 || y >= H) continue;          /* zero pad */
                        for (int kx = 0; kx < K; kx++) {
                            const int x = ox * S + kx - pad;
                            if (x < 0 || x >= H) continue;      /* zero pad */
                            acc += (acc_t)in[(ch * H + y) * H + x] *
                                   (acc_t)w[((m * Ng + n) * K + ky) * K + kx];
                        }
                    }
                }
                out[(m * Ho + oy) * Ho + ox] = requant(acc, shift[m], 1);
            }
        }
    }
}

void ref_avgpool(const data_t *in, int C, int H, data_t *out)
{
    for (int c = 0; c < C; c++) {
        long s = 0;
        for (int p = 0; p < H * H; p++) s += in[c * H * H + p];
        out[c] = (data_t)((s + H * H / 2) / (H * H));
    }
}

void ref_fc(const data_t *in, int N, const wgt_t *w, const bias_t *b, int M,
            acc_t *logits)
{
    for (int m = 0; m < M; m++) {
        acc_t acc = b[m];
        for (int n = 0; n < N; n++) acc += (acc_t)in[n] * (acc_t)w[m * N + n];
        logits[m] = acc;
    }
}

/* Count the packed parameter sizes by walking the layer table. */
void ref_param_totals(int *w_total, int *b_total)
{
    int cin = IMG_CH, w = 0, b = 0;
    for (int i = 0; i < NUM_DEFS; i++) {
        const int d = scaled_depth(MOBILENET_DEFS[i].depth);
        if (MOBILENET_DEFS[i].type == LT_CONV) {
            w += d * cin * 9;  b += d;
        } else {
            w += cin * 9;      b += cin;   /* depthwise */
            w += cin * d;      b += d;     /* pointwise */
        }
        cin = d;
    }
    w += cin * NUM_CLASSES;  b += NUM_CLASSES;
    *w_total = w;  *b_total = b;
}

/* Full network. If 'trace' is non-NULL, each conv layer's output is passed
 * to it so the testbench can print statistics. */
void ref_mobilenet(const data_t *image, const wgt_t *weights,
                   const bias_t *biases, const int8_t *shifts,
                   acc_t *logits, data_t *buf_a, data_t *buf_b,
                   ref_trace_fn trace)
{
    const wgt_t  *w = weights;
    const bias_t *b = biases;
    int H = IMG_SIZE, cin = IMG_CH, L = 0;

    /* copy the image so the loop can treat every layer the same way */
    for (int i = 0; i < IMG_CH * IMG_SIZE * IMG_SIZE; i++) buf_a[i] = image[i];
    data_t *in = buf_a, *out = buf_b, *t;

    for (int i = 0; i < NUM_DEFS; i++) {
        const int d = scaled_depth(MOBILENET_DEFS[i].depth);
        const int s = MOBILENET_DEFS[i].stride;
        const int Ho = out_size(H, s);

        if (MOBILENET_DEFS[i].type == LT_CONV) {
            ref_conv2d(in, cin, H, w, b, d, 3, s, 1, shifts + (b - biases), out);;
            if (trace) trace(L, "conv", out, d * Ho * Ho);
            L++; w += d * cin * 9; b += d;
            t = in; in = out; out = t;
        } else {
            /* depthwise: groups = cin */
            ref_conv2d(in, cin, H, w, b, cin, 3, s, cin, shifts + (b - biases), out);;
            if (trace) trace(L, "dw", out, cin * Ho * Ho);
            L++; w += cin * 9; b += cin;
            t = in; in = out; out = t;
            /* pointwise: K = 1 */
            ref_conv2d(in, cin, Ho, w, b, d, 1, 1, 1, shifts + (b - biases), out);
            if (trace) trace(L, "pw", out, d * Ho * Ho);
            L++; w += cin * d; b += d;
            t = in; in = out; out = t;
        }
        H = Ho; cin = d;
    }
    ref_avgpool(in, cin, H, out);
    ref_fc(out, cin, w, b, NUM_CLASSES, logits);
}
