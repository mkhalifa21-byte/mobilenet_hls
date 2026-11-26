#!/bin/bash
# run_250.sh -- try the v5 accelerator at 250 MHz (4 ns clock).
#
# Runs everything in order and stops at the first problem:
#   1. makes script_250.tcl from script.tcl (new project mobilenet_hls_250,
#      solution opt_250MHz, clock 4 ns). The 100 MHz project is not touched.
#   2. Vitis HLS: C simulation + synthesis + IP export        (~1-5 min)
#      -> stops here if C sim fails or the HLS Fmax estimate is below 250 MHz
#   3. Vivado: block design + bitstream at 250 MHz, 1 job      (~2-3 hours)
#   4. if timing is met: packs board_pkg_250.tgz for the board
#
# Limits: every tool runs under a 10 GB memory cap (Linux stops the tool,
# never other programs), Vivado uses 1 job x 2 threads.
#
# Start (as user mido, from ~/mobilenet_hls):
#   nohup bash run_250.sh > /dev/null 2>&1 &
# Progress:   cat run_250_status.txt
# Stop:       pkill -u mido -f run_250.sh; pkill -u mido -f vivado; pkill -u mido -f vitis_hls

cd ~/mobilenet_hls || exit 1
ST=run_250_status.txt
CAP="systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0"
say() { echo "$*" >> $ST; }
cleanup() {   # leftover tool processes (Vivado can leave child jobs behind)
    pkill -u "$USER" -f "unwrapped/lnx64.o/vivado"    2>/dev/null
    pkill -u "$USER" -f "unwrapped/lnx64.o/vitis_hls" 2>/dev/null
}

echo "started $(date)" > $ST

# ---- 0. checks -----------------------------------------------------------
if ! $CAP true > /dev/null 2>&1; then
    say "STOPPED: the 10 GB memory cap does not work in this terminal."
    say "         Log in as mido with ssh (e.g. ssh mido@localhost) and start again."
    exit 1
fi
if pgrep -u "$USER" -f "unwrapped/lnx64.o/(vivado|vitis_hls)" > /dev/null; then
    say "STOPPED: another Vivado/Vitis is already running (pgrep -a vivado)."
    exit 1
fi

# ---- 1. HLS script for 250 MHz ------------------------------------------
cp script.tcl script_250.tcl
sed -i 's/open_project -reset mobilenet_hls\b/open_project -reset mobilenet_hls_250/; s/-period 10\b/-period 4/; s/baseline_100MHz/opt_250MHz/' script_250.tcl
if ! grep -q "open_project -reset mobilenet_hls_250" script_250.tcl \
 || ! grep -qE "create_clock -period 4( |$)" script_250.tcl \
 || ! grep -q "opt_250MHz" script_250.tcl \
 || ! grep -q "^[[:space:]]*export_design" script_250.tcl; then
    say "STOPPED: script_250.tcl was not changed as expected:"
    grep -n "open_project\|open_solution\|create_clock\|export_design" script_250.tcl >> $ST
    exit 1
fi
say "script_250.tcl ready (clock 4 ns = 250 MHz)"

# ---- 2. Vitis HLS --------------------------------------------------------
say "HLS started $(date)"
$CAP vitis_hls -f script_250.tcl > vitis_250.log 2>&1
rc=$?
cleanup
RPT=mobilenet_hls_250/opt_250MHz/syn/report/csynth.rpt   # copied for the slides
if [ $rc -ne 0 ] || [ ! -f mobilenet_hls_250/opt_250MHz/impl/ip/component.xml ]; then
    say "STOPPED: HLS failed (exit code $rc) $(date). See vitis_250.log"
    grep -n "ERROR\|FAIL\|Killed" vitis_250.log | head -10 >> $ST
    exit 1
fi
mkdir -p reports && cp $RPT reports/opt_v5_250MHz.rpt
if ! grep -q "ALL TESTS PASSED" vitis_250.log; then
    say "STOPPED: C simulation did not print ALL TESTS PASSED. See vitis_250.log"
    exit 1
fi
fmax=$(grep -m1 "Estimated Fmax" vitis_250.log | grep -oE "[0-9]+(\.[0-9]+)?" | tail -1)
say "HLS done $(date): C sim passed, estimated Fmax ${fmax} MHz (target 250 MHz)"
if [ -z "$fmax" ] || ! awk -v f="$fmax" 'BEGIN{exit !(f>=250)}'; then
    say "STOPPED: HLS Fmax estimate '${fmax}' MHz is below 250 MHz (or missing), so Vivado would fail."
    exit 1
fi

# ---- 3. Vivado -----------------------------------------------------------
rm -rf vivado_prj_250
say "Vivado started $(date)"
timeout 6h $CAP vivado -mode batch -source build_bd_250.tcl -log vivado_250.log -journal vivado_250.jou > /dev/null 2>&1
rc=$?
cleanup
grep "pl_clk0 requested\|WNS" vivado_250.log | tail -2 >> $ST
if [ $rc -eq 2 ]; then
    say "STOPPED: bitstream built but timing NOT met at 250 MHz $(date). Keep using the 100 MHz one."
    exit 1
fi
if [ $rc -ne 0 ] || [ ! -f overlay_250/mobilenet.bit ]; then
    say "STOPPED: Vivado failed (exit code $rc) $(date). See vivado_250.log"
    grep -n "ERROR\|Killed" vivado_250.log | head -10 >> $ST
    exit 1
fi
cp overlay_250/timing.rpt reports/vivado_timing_250MHz.rpt
cp overlay_250/utilization.rpt reports/vivado_utilization_250MHz.rpt

# ---- 4. board package ----------------------------------------------------
rm -rf board_pkg_250 && mkdir board_pkg_250
cp overlay_250/mobilenet.bit overlay_250/mobilenet.hwh board/pynq_run.py \
   weights.bin biases.bin shifts.bin image.bin logits_ref.bin tf_logits.bin board_pkg_250/
tar czf board_pkg_250.tgz board_pkg_250
say "DONE $(date): timing met at 250 MHz. Board package: board_pkg_250.tgz"
