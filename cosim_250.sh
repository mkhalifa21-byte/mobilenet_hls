#!/bin/bash
# cosim_250.sh -- C/RTL co-simulation of the 250 MHz (v6) design.
#
# Runs the RTL that HLS generated (not the C code) on the testbench with the
# real weights and image, checks it is bit-exact, and has Vitis report the
# measured latency in clock cycles for one full image.
#
# Uses the existing project mobilenet_hls_250/opt_250MHz (made by run_250.sh);
# it does not re-synthesize and does not touch any bitstream.
# Limits: 10 GB memory cap, stops by itself after 8 hours.
#
# Start (as user mido, from ~/mobilenet_hls):
#   nohup bash cosim_250.sh > /dev/null 2>&1 &
# Progress:   cat cosim_250_status.txt
# Stop:       pkill -u mido -f cosim_250.sh; pkill -u mido -f vitis_hls; pkill -u mido -f xsim

cd ~/mobilenet_hls || exit 1
ST=cosim_250_status.txt
CAP="systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0"
say() { echo "$*" >> $ST; }
echo "started $(date)" > $ST

if [ ! -d mobilenet_hls_250/opt_250MHz/syn ]; then
    say "STOPPED: mobilenet_hls_250/opt_250MHz not found. Run run_250.sh first."; exit 1
fi
if ! $CAP true > /dev/null 2>&1; then
    say "STOPPED: the 10 GB memory cap does not work in this terminal (log in with ssh as mido)."; exit 1
fi
if pgrep -u "$USER" -f "unwrapped/lnx64.o/(vivado|vitis_hls)" > /dev/null; then
    say "STOPPED: another Vivado/Vitis is already running."; exit 1
fi

cat > cosim_250.tcl <<'EOF'
open_project mobilenet_hls_250
open_solution opt_250MHz
cosim_design -rtl verilog -trace_level none
exit
EOF

say "co-simulation started $(date) (expect roughly 1-3 hours)"
timeout 8h $CAP vitis_hls -f cosim_250.tcl > cosim_250.log 2>&1
rc=$?
pkill -u "$USER" -f "unwrapped/lnx64.o/vitis_hls" 2>/dev/null
pkill -u "$USER" -f xsim 2>/dev/null

RPT=$(ls mobilenet_hls_250/opt_250MHz/sim/report/*cosim.rpt 2>/dev/null | head -1)
if [ $rc -ne 0 ] || [ -z "$RPT" ]; then
    say "STOPPED: co-simulation failed (exit code $rc) $(date). See cosim_250.log"
    grep -n "ERROR\|FAIL\|mismatch\|Killed" cosim_250.log | head -10 >> $ST
    exit 1
fi
mkdir -p reports && cp "$RPT" reports/cosim_250MHz.rpt
grep -q "ALL TESTS PASSED" cosim_250.log && say "testbench: ALL TESTS PASSED (RTL output bit-exact)" \
                                         || say "WARNING: ALL TESTS PASSED not found in cosim_250.log"
say "DONE $(date). Report: reports/cosim_250MHz.rpt"
grep -A6 "RTL" reports/cosim_250MHz.rpt | head -8 >> $ST
