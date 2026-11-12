# script.tcl -- Vitis HLS 2023.1 batch flow for the BASELINE design
#
# On balboa / doheny:
#   source /ecelib/eceware/xilinx_2023.1/Vitis/2023.1/settings64.csh
#   vitis_hls -f script.tcl
#
# Part: XCK26 (the K26 SOM used on both the KR260 and KV260 boards)

open_project -reset mobilenet_hls
set_top mobilenet_top

add_files mobilenet_kernels.cpp
add_files mobilenet_top.cpp
add_files -tb tb_mobilenet.cpp
add_files -tb mobilenet_ref.cpp
add_files -tb mobilenet_ref.h
# When you have trained weights, uncomment so csim can find them:
# add_files -tb weights.bin
# add_files -tb biases.bin
# add_files -tb shifts.bin
# add_files -tb image.bin

open_solution -reset "baseline_100MHz" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default   ;# 10 ns = 100 MHz

csim_design          ;# 1. C simulation: must print ALL TESTS PASSED
csynth_design        ;# 2. C synthesis: report in solution/syn/report/
# cosim_design       ;# 3. Full-network RTL co-sim takes many hours; do it
#                    ;#    per kernel once the optimized kernels exist
# export_design -format ip_catalog   ;# 4. IP for the Vivado block design

exit
