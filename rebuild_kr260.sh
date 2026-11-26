#!/bin/bash
# rebuild_kr260.sh -- rebuild both bitstreams for the KR260 board.
#
# The HLS accelerators do not change (same K26 chip). Only the Vivado block
# design is rebuilt with the KR260 board preset instead of KV260:
#   1. 250 MHz (v6 IP, mobilenet_hls_250)       -> overlay_250_kr260/
#   2. 100 MHz (v5 IP, backup, mobilenet_hls)   -> overlay_100_kr260/
# Each ~20-40 min, one after the other. The KV260 builds are not touched.
#
# Limits: 10 GB memory cap, 1 Vivado job x 2 threads, 4-hour timeout per build.
#
# Start (as user mido, from ~/mobilenet_hls):
#   nohup bash rebuild_kr260.sh > /dev/null 2>&1 &
# Progress:   cat kr260_status.txt
# Stop:       pkill -u mido -f rebuild_kr260.sh; pkill -u mido -f vivado

cd ~/mobilenet_hls || exit 1
ST=kr260_status.txt
CAP="systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0"
say() { echo "$*" >> $ST; }
cleanup() { pkill -u "$USER" -f "unwrapped/lnx64.o/vivado" 2>/dev/null; }
echo "started $(date)" > $ST

# ---- checks --------------------------------------------------------------
if ! $CAP true > /dev/null 2>&1; then
    say "STOPPED: the 10 GB memory cap does not work in this terminal (log in with ssh as mido)."; exit 1
fi
if pgrep -u "$USER" -f "unwrapped/lnx64.o/(vivado|vitis_hls)" > /dev/null; then
    say "STOPPED: another Vivado/Vitis is already running."; exit 1
fi
for ip in mobilenet_hls_250/opt_250MHz/impl/ip mobilenet_hls/baseline_100MHz/impl/ip; do
    if [ ! -f $ip/component.xml ]; then say "STOPPED: HLS IP missing: $ip"; exit 1; fi
done

# keep the new build folders out of git
for p in 'vivado_prj_*_kr260/' 'overlay_*_kr260/' 'build_kr260_*.tcl' 'kr260_status.txt'; do
    grep -qxF "$p" .gitignore || echo "$p" >> .gitignore
done

# ---- Vivado scripts for KR260 (made from build_bd_250.tcl) ---------------
sed -e 's/^set BOARD    "kv260"/set BOARD    "kr260"/' \
    -e 's#^set PRJ_DIR .*#set PRJ_DIR  [file normalize "vivado_prj_250_kr260"]#' \
    -e 's#^set OUT_DIR .*#set OUT_DIR  [file normalize "overlay_250_kr260"]#' \
    build_bd_250.tcl > build_kr260_250.tcl
sed -e 's/^set BOARD    "kv260"/set BOARD    "kr260"/' \
    -e 's/^set CLK_MHZ  250/set CLK_MHZ  100/' \
    -e 's#^set HLS_IP .*#set HLS_IP   [file normalize "mobilenet_hls/baseline_100MHz/impl/ip"]#' \
    -e 's#^set PRJ_DIR .*#set PRJ_DIR  [file normalize "vivado_prj_100_kr260"]#' \
    -e 's#^set OUT_DIR .*#set OUT_DIR  [file normalize "overlay_100_kr260"]#' \
    build_bd_250.tcl > build_kr260_100.tcl
for f in build_kr260_250.tcl build_kr260_100.tcl; do
    if ! grep -q '^set BOARD    "kr260"' $f || ! grep -q '^set JOBS     1' $f; then
        say "STOPPED: $f was not generated as expected"; exit 1
    fi
done
grep -q '^set CLK_MHZ  100' build_kr260_100.tcl || { say "STOPPED: 100 MHz clock not set"; exit 1; }

# ---- build one bitstream --------------------------------------------------
build() {   # $1 = 250 or 100
    local mhz=$1 tcl=build_kr260_$1.tcl prj=vivado_prj_$1_kr260 out=overlay_$1_kr260
    rm -rf $prj
    say "[$mhz MHz] Vivado started $(date)"
    timeout 4h $CAP vivado -mode batch -source $tcl -log vivado_kr260_$mhz.log \
        -journal vivado_kr260_$mhz.jou > /dev/null 2>&1
    local rc=$?
    cleanup
    grep "using board part\|pl_clk0 requested\|worst setup slack" vivado_kr260_$mhz.log | grep -v "^#" >> $ST
    if [ $rc -eq 2 ]; then say "[$mhz MHz] STOPPED: timing NOT met $(date)"; return 1; fi
    if [ $rc -ne 0 ] || [ ! -f $out/mobilenet.bit ]; then
        say "[$mhz MHz] STOPPED: Vivado failed (exit code $rc) $(date). See vivado_kr260_$mhz.log"
        grep -n "ERROR\|Killed" vivado_kr260_$mhz.log | head -8 >> $ST
        return 1
    fi
    mkdir -p reports
    cp $out/timing.rpt reports/vivado_timing_${mhz}MHz_kr260.rpt
    cp $out/utilization.rpt reports/vivado_utilization_${mhz}MHz_kr260.rpt
    local pkg=board_pkg_kr260_$mhz
    rm -rf $pkg && mkdir $pkg
    cp $out/mobilenet.bit $out/mobilenet.hwh board/pynq_run.py \
       weights.bin biases.bin shifts.bin image.bin logits_ref.bin tf_logits.bin $pkg/
    tar czf $pkg.tgz $pkg
    say "[$mhz MHz] DONE $(date): timing met. Board package: $pkg.tgz"
}

build 250
build 100
say "finished $(date)"
