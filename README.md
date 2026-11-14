# MobileNetV1 on Kria K26 — HLS baseline

EECS 298 final project. This is a hand translation of TF-slim `mobilenet_v1.py`
into C for Vitis HLS: MobileNetV1, width multiplier α = 0.5, 160×160 input,
1001 classes.

This is the **baseline**. It has no performance pragmas, and every access goes
to DDR. Synthesize it first and record the numbers. Each optimization after
that has to keep `ALL TESTS PASSED`.

## Files

| File | Synthesized? | What it is |
|---|---|---|
| `mobilenet.h` | yes | Types, number format, layer table (= `MOBILENETV1_CONV_DEFS`), SAME padding, rounding |
| `mobilenet_kernels.cpp` | yes | `conv3x3`, `dwconv3x3`, `conv1x1`, `avgpool`, `fc`: naive loop nests in the Lecture 2 style |
| `mobilenet_top.cpp` | yes | `mobilenet_top()`: runs all 28 layers, with the m_axi / s_axilite interfaces |
| `mobilenet_ref.cpp/.h` | no (testbench) | Golden reference: one generic grouped conv used for all layer types |
| `tb_mobilenet.cpp` | no (testbench) | Kernel unit tests plus a full-network test, all bit-exact |
| `script.tcl` | | Vitis HLS 2023.1 flow (csim + csynth) |
| `Makefile` | | Quick g++ build: `make run` |
| `python/export_weights.py` | no (run once, offline) | Converts the trained TF checkpoint into the `.bin` files |

## Python → C mapping

| mobilenet_v1.py | C |
|---|---|
| `MOBILENETV1_CONV_DEFS` (l.123–138) | `MOBILENET_DEFS[]` in `mobilenet.h` |
| `depth = max(int(d*α), 8)` (l.217) | `scaled_depth()` |
| `slim.conv2d 3×3 stride 2` (Conv2d_0) | `conv3x3()` |
| `slim.separable_conv2d(net, None, …)` | `dwconv3x3()` |
| `slim.conv2d [1,1]` | `conv1x1()` |
| `batch_norm` (ε = 0.001) | folded into weights + bias offline |
| `relu6` | the clamp to [0, 96] in `requant()` |
| `padding='SAME'` | `same_pad_begin()`: stride 2 pads only bottom/right |
| `avg_pool2d 7×7` | `avgpool()` |
| `Conv2d_1c_1x1` logits | `fc()` |

## Number format

- **Feature maps:** int8 Q4.4. Real value = stored ÷ 16. 6.0 is stored as 96.
- **Weights:** int8, with a per-layer shift: real value = stored ÷ 2^shift[layer].
- **Biases:** int32 at the accumulator scale. Real value = stored ÷ 2^(4 + shift).
- **Layer output:** `clamp((acc + 2^(shift−1)) >> shift, 0, 96)`.

Because everything is integer, C simulation, RTL simulation and the board must
produce identical numbers. That's why the testbench requires an exact match.

## Running

```sh
make run                      # g++, about 1 s
vitis_hls -f script.tcl       # on balboa/doheny after sourcing settings64.csh
```

Expected output ends with `*** ALL TESTS PASSED ***`. The testbench also prints
per-layer statistics (mean, % zero, % clipped at 6.0), which show that the data
is non-trivial at every layer.

## Weights

- **Default:** deterministic random weights (He-style scale). Latency,
  resources and correctness are identical to using trained weights. The
  predicted class is meaningless.
- **Trained weights** (approved by the professor: Python is used once,
  offline, only to export them):

  ```sh
  cd python
  pip3 install -r requirements.txt          # tensorflow, numpy, pillow
  # download mobilenet_v1_0.50_160.tgz from the "Pre-trained Models" table in
  # the tensorflow/models repo (research/slim/README.md) and unpack it here
  python3 export_weights.py --ckpt mobilenet_v1_0.5_160.ckpt --size 160 \
                            --image some_photo.jpg
  ```

  This writes `weights.bin`, `biases.bin`, `shifts.bin`, `image.bin`,
  `tf_logits.bin` and `export_report.txt` into the project folder. Then run
  `make run` again. The testbench loads them automatically and also prints
  how closely the int8 model agrees with the original float TensorFlow model
  (top-1 match and score correlation; above 0.95 is good).

  `export_weights.py` only reads the checkpoint with plain TensorFlow. It does
  not need `tf_slim`, which no longer works with current TensorFlow. The
  variable names it reads are the ones `mobilenet_v1.py` creates.

  `--size` must equal `IMG_SIZE` in `mobilenet.h` (currently 160).

## Baseline numbers to record (from `solution/syn/report/`)

- Latency (cycles) of `mobilenet_top` and of each kernel
- II of each innermost loop
- LUT / FF / DSP / BRAM / URAM usage
- Estimated clock period

Repeat with `create_clock -period 6.67` (150 MHz) and `-period 4` (250 MHz).

## Next: optimizations (each must still pass the testbench)

1. `conv1x1`: tile `Tm × Tn`, keep weight and input tiles in on-chip arrays,
   `ARRAY_PARTITION` + `UNROLL` the tile loops, `PIPELINE` the pixel loop.
   About 90% of all MACs are here.
2. `dwconv3x3`: line buffer (2 rows) + 3×3 window registers, 16+ channels in
   parallel, all 9 MACs per cycle.
3. Move feature maps from DDR to on-chip URAM (`BIND_STORAGE ... impl=uram`).
4. Widen the m_axi ports to 128 bits.
