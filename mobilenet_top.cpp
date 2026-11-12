/*
 * mobilenet_top.cpp  --  HLS top function: runs the whole network
 *
 * Direct translation of the loop in mobilenet_v1_base() (lines 246-297)
 * plus the Logits block in mobilenet_v1() (lines 367-387).
 *
 * DDR usage (the ARM host allocates these and passes their addresses):
 *   image    input picture, Q4.4, [3][224][224]
 *   fm       scratch area of 3 regions (A, B, TMP), each FM_MAX bytes.
 *            Layers ping-pong between A and B ("buffer 1 / buffer 2" in the
 *            Lecture 2 system diagram); depthwise output goes to TMP.
 *   weights / biases / shifts   packed parameters, see mobilenet.h
 *   logits   1001 raw scores; the predicted class is the largest one.
 *
 * Interface pragmas use the same style as the course's vecadd_buffer.cpp.
 */
#include "mobilenet.h"

void mobilenet_top(const data_t *image, data_t *fm, const wgt_t *weights,
                   const bias_t *biases, const int8_t *shifts, acc_t *logits)
{
#pragma HLS INTERFACE m_axi port=image   offset=slave bundle=gmem0 depth=150528
#pragma HLS INTERFACE m_axi port=fm      offset=slave bundle=gmem1 depth=1204224
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem2 depth=1320160
#pragma HLS INTERFACE m_axi port=biases  offset=slave bundle=gmem3 depth=6473
#pragma HLS INTERFACE m_axi port=shifts  offset=slave bundle=gmem3 depth=27
#pragma HLS INTERFACE m_axi port=logits  offset=slave bundle=gmem0 depth=1001
#pragma HLS INTERFACE s_axilite register port=image   bundle=ctrl
#pragma HLS INTERFACE s_axilite register port=fm      bundle=ctrl
#pragma HLS INTERFACE s_axilite register port=weights bundle=ctrl
#pragma HLS INTERFACE s_axilite register port=biases  bundle=ctrl
#pragma HLS INTERFACE s_axilite register port=shifts  bundle=ctrl
#pragma HLS INTERFACE s_axilite register port=logits  bundle=ctrl
#pragma HLS INTERFACE s_axilite register port=return  bundle=ctrl

    int H     = IMG_SIZE;   /* current feature-map height (= width) */
    int cin   = IMG_CH;     /* current number of channels           */
    int w_off = 0;          /* where this layer's weights start     */
    int b_off = 0;          /* where this layer's biases start      */
    int L     = 0;          /* conv layer counter (index into shifts) */
    int cur   = REGION_A;   /* region holding the current input     */

    /* ---- Conv2d_0: standard 3x3 conv, stride 2, 3 -> 16 channels ---- */
    {
        const int d = scaled_depth(MOBILENET_DEFS[0].depth);
        const int s = MOBILENET_DEFS[0].stride;
        conv3x3(image, fm + REGION_A, weights + w_off, biases + b_off,
                cin, d, H, s, shifts[L]);
        L++;
        w_off += d * cin * 9;
        b_off += d;
        H   = out_size(H, s);
        cin = d;
        cur = REGION_A;
    }

    /* ---- Conv2d_1 ... Conv2d_13: depthwise 3x3 then pointwise 1x1 ---- */
LAYERS: for (int i = 1; i < NUM_DEFS; i++) {
        const int d  = scaled_depth(MOBILENET_DEFS[i].depth);
        const int s  = MOBILENET_DEFS[i].stride;
        const int Ho = out_size(H, s);
        const int nxt = (cur == REGION_A) ? REGION_B : REGION_A;

        /* Conv2d_i_depthwise: cin -> cin channels, H -> Ho */
        dwconv3x3(fm + cur, fm + REGION_TMP, weights + w_off, biases + b_off,
                  cin, H, s, shifts[L]);
        L++;
        w_off += cin * 9;
        b_off += cin;

        /* Conv2d_i_pointwise: cin -> d channels, size stays Ho */
        conv1x1(fm + REGION_TMP, fm + nxt, weights + w_off, biases + b_off,
                cin, d, Ho, shifts[L]);
        L++;
        w_off += cin * d;
        b_off += d;

        cur = nxt;
        H   = Ho;
        cin = d;
    }

    /* ---- Logits: 7x7 average pool, then 1x1 conv to 1001 classes ---- */
    avgpool(fm + cur, fm + REGION_TMP, cin, H);
    fc(fm + REGION_TMP, logits, weights + w_off, biases + b_off,
       cin, NUM_CLASSES);
}
