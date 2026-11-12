/* mobilenet_ref.h -- golden reference (testbench only) */
#ifndef MOBILENET_REF_H
#define MOBILENET_REF_H

#include "mobilenet.h"

typedef void (*ref_trace_fn)(int layer, const char *kind,
                             const data_t *out, int count);

void ref_conv2d(const data_t *in, int N, int H,
                const wgt_t *w, const bias_t *b, int M, int K, int S,
                int groups, int shift, data_t *out);
void ref_avgpool(const data_t *in, int C, int H, data_t *out);
void ref_fc(const data_t *in, int N, const wgt_t *w, const bias_t *b, int M,
            acc_t *logits);
void ref_param_totals(int *w_total, int *b_total);
void ref_mobilenet(const data_t *image, const wgt_t *weights,
                   const bias_t *biases, const int8_t *shifts,
                   acc_t *logits, data_t *buf_a, data_t *buf_b,
                   ref_trace_fn trace);

#endif
