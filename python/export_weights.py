#!/usr/bin/env python3
"""
export_weights.py -- convert Google's trained MobileNetV1 checkpoint into the
binary files read by the C testbench (tb_mobilenet.cpp).

Run ONCE on a PC. Python is not part of the HLS design or the hardware.

What it does, step by step
  1. Reads every layer's weights and BatchNorm values from the TF-slim
     checkpoint (variable names as created by mobilenet_v1.py).
  2. Folds BatchNorm into each conv layer, per output channel c:
         s[c]      = gamma[c] / sqrt(moving_variance[c] + eps)   (eps = 0.001)
         w_new[c]  = w[c] * s[c]
         b_new[c]  = beta[c] - moving_mean[c] * s[c]
  3. Quantizes to int8 with one power-of-two shift per OUTPUT CHANNEL
     (one shift for the whole fc layer):
         shift[c]  = floor(log2(127 / max|w_new[c]|)), limited to 0..14
         w_int8    = round(w_new[c] * 2^shift[c])
         b_int32   = round(b_new[c] * 2^(8 + shift[c]))  (8 = data frac bits)
  4. Reorders from TensorFlow's layout to the C layout in mobilenet.h:
         standard conv  TF [kh][kw][in][out]  ->  C [out][in][kh][kw]
         depthwise      TF [kh][kw][C][1]     ->  C [C][kh][kw]
         pointwise/fc   TF [1][1][in][out]    ->  C [out][in]
  5. Prepares a test image (center crop 87.5%, resize, scale to [-1, 1]),
     stored as Q8.8 int16 in [channel][row][col].
  6. Runs the ORIGINAL float model (un-folded BatchNorm, TensorFlow's own
     conv/padding ops) on the same image and saves its 1001 scores, so the
     C testbench can check that the int8 hardware model agrees with it.

Outputs (written to --out, default: the HLS project folder one level up)
  weights.bin   int8   [W_TOTAL]      all layers packed in network order
  biases.bin    int32  [B_TOTAL]
  shifts.bin    int8   [5472]         one per conv output channel
  image.bin     int16  [3*S*S]        Q8.8
  tf_logits.bin float32 [1001]        reference scores from TensorFlow
  export_report.txt                   per-layer shift and quantization error

Usage
  pip install tensorflow pillow numpy
  # download + unpack mobilenet_v1_0.5_224.tgz (or _160) from the TF-slim
  # pretrained models table, then:
  python3 export_weights.py --ckpt mobilenet_v1_0.5_224.ckpt --size 224 \
                            --image cat.jpg
"""
import argparse
import os
import sys

import numpy as np

# ---- must match mobilenet.h -------------------------------------------------
FA = 8                      # fractional bits of feature maps (Q8.8, int16)
ALPHA = 0.5
MIN_DEPTH = 8
NUM_CLASSES = 1001
BN_EPS = 0.001              # mobilenet_v1_arg_scope(batch_norm_epsilon=0.001)

# MOBILENETV1_CONV_DEFS from mobilenet_v1.py: (type, stride, depth at alpha=1)
CONV_DEFS = [("conv", 2, 32)] + [
    ("dwsep", 1, 64), ("dwsep", 2, 128), ("dwsep", 1, 128),
    ("dwsep", 2, 256), ("dwsep", 1, 256), ("dwsep", 2, 512),
    ("dwsep", 1, 512), ("dwsep", 1, 512), ("dwsep", 1, 512),
    ("dwsep", 1, 512), ("dwsep", 1, 512),
    ("dwsep", 2, 1024), ("dwsep", 1, 1024),
]


def depth(d):
    return max(int(d * ALPHA), MIN_DEPTH)


# ---- reading the checkpoint -------------------------------------------------
class Checkpoint:
    def __init__(self, path):
        import tensorflow as tf
        self.reader = tf.train.load_checkpoint(path)
        self.names = set(self.reader.get_variable_to_shape_map().keys())

    def get(self, name, required=True):
        if name in self.names:
            return self.reader.get_tensor(name).astype(np.float64)
        if required:
            close = sorted(n for n in self.names if name.split("/")[1] in n)[:10]
            sys.exit(f"ERROR: variable '{name}' not in checkpoint.\n"
                     f"Similar names: {close}\n"
                     f"Is this the alpha={ALPHA} MobileNetV1 slim checkpoint?")
        return None

    def bn(self, scope):
        """gamma, beta, mean, var of one BatchNorm layer."""
        n = len(self.get(scope + "/BatchNorm/beta"))
        gamma = self.get(scope + "/BatchNorm/gamma", required=False)
        if gamma is None:
            gamma = np.ones(n)
        return (gamma, self.get(scope + "/BatchNorm/beta"),
                self.get(scope + "/BatchNorm/moving_mean"),
                self.get(scope + "/BatchNorm/moving_variance"))


def layer_list(ck):
    """All layers in network order, with TF-layout float weights."""
    layers = []
    for i, (kind, _, _) in enumerate(CONV_DEFS):
        if kind == "conv":
            scope = f"MobilenetV1/Conv2d_{i}"
            layers.append(dict(name=f"Conv2d_{i}", kind="conv",
                               w=ck.get(scope + "/weights"), bn=ck.bn(scope)))
        else:
            scope = f"MobilenetV1/Conv2d_{i}_depthwise"
            layers.append(dict(name=f"Conv2d_{i}_depthwise", kind="dw",
                               w=ck.get(scope + "/depthwise_weights"),
                               bn=ck.bn(scope)))
            scope = f"MobilenetV1/Conv2d_{i}_pointwise"
            layers.append(dict(name=f"Conv2d_{i}_pointwise", kind="pw",
                               w=ck.get(scope + "/weights"), bn=ck.bn(scope)))
    layers.append(dict(name="Logits", kind="fc",
                       w=ck.get("MobilenetV1/Logits/Conv2d_1c_1x1/weights"),
                       b=ck.get("MobilenetV1/Logits/Conv2d_1c_1x1/biases"),
                       bn=None))
    return layers


# ---- fold, quantize, reorder ------------------------------------------------
def fold_bn(layer):
    """Return (w, b) with BatchNorm folded in; w stays in TF layout."""
    w = layer["w"]
    if layer["bn"] is None:                       # fc layer has a real bias
        return w, layer["b"]
    gamma, beta, mean, var = layer["bn"]
    s = gamma / np.sqrt(var + BN_EPS)             # one scale per out channel
    if layer["kind"] == "dw":
        w = w * s.reshape(1, 1, -1, 1)            # channel axis is 2
    else:
        w = w * s.reshape(1, 1, 1, -1)            # out-channel axis is 3
    return w, beta - mean * s


def to_c_layout(w, kind):
    if kind == "conv":
        return w.transpose(3, 2, 0, 1)            # [out][in][kh][kw]
    if kind == "dw":
        return w[:, :, :, 0].transpose(2, 0, 1)   # [C][kh][kw]
    return w[0, 0].T                              # [out][in]


def quantize(w, b, per_channel=True):
    """w in C layout: axis 0 = output channel.
    per_channel=True : one shift per output channel (conv layers)
    per_channel=False: one shift for the whole layer (fc, so all 1001
                       scores share the same scale and argmax is valid)"""
    M = w.shape[0]
    if per_channel:
        mx = np.abs(w.reshape(M, -1)).max(axis=1)          # [M]
    else:
        mx = np.full(M, np.abs(w).max())
    shift = np.floor(np.log2(127.0 / np.maximum(mx, 1e-12))).astype(int)
    if np.any(shift < 0):
        sys.exit("ERROR: a weight is > 127 after BatchNorm folding; "
                 "negative shifts are not supported")
    shift = np.clip(shift, 0, 14)
    scale = 2.0 ** shift.reshape((M,) + (1,) * (w.ndim - 1))
    wq = np.clip(np.round(w * scale), -128, 127).astype(np.int8)
    bq = np.round(b * 2.0 ** (FA + shift))
    bq = np.clip(bq, -2**31, 2**31 - 1).astype(np.int32)
    err = np.linalg.norm(wq / scale - w) / max(np.linalg.norm(w), 1e-12)
    return wq, bq, shift, float(mx.max()), err


# ---- image ------------------------------------------------------------------
def load_image(path, size):
    """Inception-style eval preprocessing used by the TF-slim models."""
    if path is None:
        print("WARNING: no --image given, using a random image "
              "(the predicted class will be meaningless)")
        rng = np.random.default_rng(0)
        return rng.uniform(-1, 1, (size, size, 3)).astype(np.float32)
    from PIL import Image
    img = Image.open(path).convert("RGB")
    W, H = img.size
    cw, ch = int(W * 0.875), int(H * 0.875)       # central_crop(0.875)
    left, top = (W - cw) // 2, (H - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))
    img = img.resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return (x - 0.5) * 2.0                        # [-1, 1], shape [S][S][3]


# ---- float reference with TensorFlow's own ops ------------------------------
def tf_reference(layers, image):
    """Original float model: real BatchNorm (not folded), TF 'SAME' padding."""
    import tensorflow as tf
    x = tf.constant(image[None].astype(np.float32))
    li = 0
    for kind, stride, _ in CONV_DEFS:
        todo = ["conv"] if kind == "conv" else ["dw", "pw"]
        for k in todo:
            L = layers[li]; li += 1
            w = tf.constant(L["w"].astype(np.float32))
            st = stride if k in ("conv", "dw") else 1
            if k == "dw":
                x = tf.nn.depthwise_conv2d(x, w, [1, st, st, 1], "SAME")
            else:
                x = tf.nn.conv2d(x, w, [1, st, st, 1], "SAME")
            g, be, m, v = (a.astype(np.float32) for a in L["bn"])
            x = (x - m) / np.sqrt(v + BN_EPS) * g + be
            x = tf.nn.relu6(x)
    x = tf.reduce_mean(x, axis=[1, 2])            # 7x7 (or 5x5) average pool
    fcl = layers[li]
    logits = tf.matmul(x, fcl["w"][0, 0].astype(np.float32)) + \
        fcl["b"].astype(np.float32)
    return logits.numpy()[0]


# ---- main -------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint prefix, e.g. mobilenet_v1_0.5_224.ckpt")
    ap.add_argument("--size", type=int, default=224, choices=[128, 160, 192, 224],
                    help="input size; must equal IMG_SIZE in mobilenet.h")
    ap.add_argument("--image", default=None, help="test photo (jpg/png)")
    ap.add_argument("--out", default=os.path.dirname(here),
                    help="output folder (default: the HLS project folder)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = Checkpoint(args.ckpt)
    layers = layer_list(ck)

    all_w, all_b, shifts, report = [], [], [], []
    report.append(f"{'layer':24s} {'shape (C layout)':>18s} {'max|w|':>8s} "
                  f"{'shifts':>5s} {'w error':>8s}")
    for L in layers:
        w, b = fold_bn(L)
        w = to_c_layout(w, L["kind"])
        is_fc = L["kind"] == "fc"
        wq, bq, sh, mx, err = quantize(w, b, per_channel=not is_fc)
        all_w.append(wq.ravel()); all_b.append(bq.ravel())
        if not is_fc:
            shifts.extend(int(x) for x in sh)
        flag = "  <-- high error" if err > 0.05 else ""
        report.append(f"{L['name']:24s} {str(wq.shape):>18s} {mx:8.3f} "
                      f"{sh.min():2d}-{sh.max():<2d} {100*err:7.2f}%{flag}")

    weights = np.concatenate(all_w)
    biases = np.concatenate(all_b)
    assert len(shifts) == 5472, len(shifts)        # S_TOTAL in mobilenet.h

    image = load_image(args.image, args.size)
    img_q = np.clip(np.round(image * 2 ** FA), -32768, 32767).astype("<i2")
    img_q = img_q.transpose(2, 0, 1)              # [3][S][S]

    weights.tofile(os.path.join(args.out, "weights.bin"))
    biases.astype("<i4").tofile(os.path.join(args.out, "biases.bin"))
    np.array(shifts, dtype=np.int8).tofile(os.path.join(args.out, "shifts.bin"))
    img_q.tofile(os.path.join(args.out, "image.bin"))

    logits = tf_reference(layers, image)
    logits.astype("<f4").tofile(os.path.join(args.out, "tf_logits.bin"))
    top5 = np.argsort(-logits)[:5]

    report.append("")
    report.append(f"weights: {weights.size} int8   (W_TOTAL in mobilenet.h)")
    report.append(f"biases : {biases.size} int32  (B_TOTAL in mobilenet.h)")
    report.append(f"image  : {args.size}x{args.size}  "
                  f"(IMG_SIZE in mobilenet.h must match)")
    report.append(f"TensorFlow float top-5 classes: {list(map(int, top5))}")
    text = "\n".join(report)
    print(text)
    with open(os.path.join(args.out, "export_report.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nWrote weights.bin biases.bin shifts.bin image.bin tf_logits.bin "
          f"to {args.out}")


if __name__ == "__main__":
    main()
