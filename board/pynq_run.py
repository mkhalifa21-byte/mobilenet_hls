#!/usr/bin/env python3
"""
pynq_run.py -- run the MobileNetV1 accelerator on the Kria board (PYNQ).

Put these files in ONE folder on the board:
    mobilenet.bit, mobilenet.hwh        from overlay/ (build_bd.tcl)
    weights.bin, biases.bin, shifts.bin,
    image.bin, tf_logits.bin            from export_weights.py
    logits_ref.bin                      from make run (C model's exact output)
    pynq_run.py                         this file

Run (PYNQ needs root):
    sudo -E python3 pynq_run.py            # or from a Jupyter notebook
Options:
    --runs N     number of timed runs (default 10)

What it does:
    1. loads the bitstream (programs the FPGA)
    2. copies weights + image into DDR buffers the accelerator can reach
    3. writes the buffer addresses into the accelerator's registers
    4. starts it, waits for it to finish, and times it
    5. checks the 1001 scores against the C model (must match exactly)
       and against TensorFlow (top-1 / correlation)
"""
import argparse
import os
import time

import numpy as np
from pynq import Overlay, allocate

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- must match mobilenet.h -------------------------------------------------
IMG_SIZE    = 160
FM_MAX      = (IMG_SIZE // 2) * (IMG_SIZE // 2) * 32      # 204,800
FM_TOTAL    = 3 * FM_MAX                                  # 614,400
W_TOTAL     = 1320160
B_TOTAL     = 6473
S_TOTAL     = 5472
NUM_CLASSES = 1001
FA          = 8                                           # Q8.8

# Control register bits (Vitis HLS ap_ctrl_hs, offset 0x00)
AP_START = 0x1
AP_IDLE  = 0x4


def path(name):
    return os.path.join(HERE, name)


def load_bin(name, dtype, count):
    a = np.fromfile(path(name), dtype=dtype)
    if a.size != count:
        raise SystemExit(f"{name}: expected {count} values, found {a.size}")
    return a


def set_pointer(ip, name, buf):
    """Write a buffer's physical address into the accelerator register.
    PYNQ shows a 64-bit pointer either as one register (name) or as two
    32-bit halves (name_1 = low, name_2 = high), depending on the version."""
    addr = buf.device_address
    rm = ip.register_map
    try:
        setattr(rm, name, addr)
    except (AttributeError, ValueError):
        setattr(rm, name + "_1", addr & 0xFFFFFFFF)
        setattr(rm, name + "_2", (addr >> 32) & 0xFFFFFFFF)


def run_once(ip):
    ip.write(0x00, AP_START)
    while (ip.read(0x00) & AP_IDLE) == 0:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    # 1. program the FPGA
    t0 = time.perf_counter()
    ol = Overlay(path("mobilenet.bit"))
    print(f"Bitstream loaded in {time.perf_counter() - t0:.2f} s")
    ip = ol.mobilenet_top_0
    print("Accelerator registers:")
    print(ip.register_map)

    # 2. DDR buffers the accelerator can read/write (physically contiguous)
    image   = allocate(shape=(3 * IMG_SIZE * IMG_SIZE,), dtype=np.int16)
    fm      = allocate(shape=(FM_TOTAL,), dtype=np.int16)
    weights = allocate(shape=(W_TOTAL,), dtype=np.int8)
    biases  = allocate(shape=(B_TOTAL,), dtype=np.int32)
    shifts  = allocate(shape=(S_TOTAL,), dtype=np.int8)
    logits  = allocate(shape=(NUM_CLASSES,), dtype=np.int32)

    image[:]   = load_bin("image.bin",   np.int16, 3 * IMG_SIZE * IMG_SIZE)
    weights[:] = load_bin("weights.bin", np.int8,  W_TOTAL)
    biases[:]  = load_bin("biases.bin",  np.int32, B_TOTAL)
    shifts[:]  = load_bin("shifts.bin",  np.int8,  S_TOTAL)
    for b in (image, weights, biases, shifts):
        b.flush()                      # make sure the data is in DDR

    # 3. buffer addresses -> accelerator registers
    #    ('image' is called 'image_r' by HLS because 'image' is reserved)
    set_pointer(ip, "image_r", image)
    set_pointer(ip, "fm",      fm)
    set_pointer(ip, "weights", weights)
    set_pointer(ip, "biases",  biases)
    set_pointer(ip, "shifts",  shifts)
    set_pointer(ip, "logits",  logits)

    # 4. warm-up run, then timed runs
    run_once(ip)
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        run_once(ip)
        times.append(time.perf_counter() - t0)
    logits.invalidate()                # read fresh results from DDR
    out = np.array(logits, dtype=np.int64)

    t = np.array(times) * 1000.0
    print(f"\nLatency over {args.runs} runs: mean {t.mean():.2f} ms, "
          f"min {t.min():.2f} ms, max {t.max():.2f} ms")
    print(f"Throughput: {1000.0 / t.mean():.1f} FPS")
    ops = 153_063_424                  # 2 x 76,525,312 MACs + avgpool adds
    print(f"Effective: {ops / (t.mean() / 1000.0) / 1e9:.2f} GOPS")

    # 5. correctness
    top5 = np.argsort(-out)[:5]
    print(f"\nBoard top-5 classes: {list(map(int, top5))}")

    if os.path.exists(path("logits_ref.bin")):
        ref = load_bin("logits_ref.bin", np.int32, NUM_CLASSES).astype(np.int64)
        mism = int(np.sum(out != ref))
        print(f"vs C model (logits_ref.bin): {mism} mismatching scores "
              f"-> {'BIT-EXACT PASS' if mism == 0 else 'FAIL'}")
    else:
        print("logits_ref.bin not found: skipping bit-exact check")

    if os.path.exists(path("tf_logits.bin")):
        tf = load_bin("tf_logits.bin", np.float32, NUM_CLASSES)
        r = np.corrcoef(out.astype(np.float64), tf.astype(np.float64))[0, 1]
        tf_top5 = np.argsort(-tf)[:5]
        print(f"vs TensorFlow: TF top-5 {list(map(int, tf_top5))}, "
              f"top-1 {'agrees' if tf_top5[0] == top5[0] else 'DIFFERS'}, "
              f"correlation {r:.4f}")

    for b in (image, fm, weights, biases, shifts, logits):
        b.freebuffer()


if __name__ == "__main__":
    main()
