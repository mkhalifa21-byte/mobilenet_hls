
# build_bd.tcl -- Vivado 2023.1: block design + bitstream for the Kria board
#
# What it builds:
#   Zynq UltraScale+ processor (PS)  --M_AXI_HPM0_FPD-->  mobilenet_top s_axi_ctrl
#   mobilenet_top m_axi_gmem0..3     --S_AXI_HP0_FPD -->  PS DDR memory
#   100 MHz PL clock (pl_clk0) + reset for the accelerator
#
# Before running:
#   1. In script.tcl, after csynth_design, have:  export_design -format ip_catalog
#      and run  vitis_hls -f script.tcl  (creates the IP in
#      mobilenet_hls/baseline_100MHz/impl/ip)
#   2. Set BOARD below to your board: kv260 or kr260 (same K26 chip; this
#      only picks the matching processor presets).
#
# Run from ~/mobilenet_hls (roughly 30-60 minutes), with a hard
# memory cap so Vivado can never take over the PC:
#   systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0 \
#     vivado -mode batch -source build_bd.tcl -log vivado.log -journal vivado.jou
#
# Output (copy this folder to the board):
#   overlay/mobilenet.bit       FPGA bitstream
#   overlay/mobilenet.hwh       hardware description PYNQ needs
#   overlay/utilization.rpt     post-implementation resources
#   overlay/timing.rpt          post-implementation timing
 
set BOARD    "kv260"
set CLK_MHZ  100
set HLS_IP   [file normalize "mobilenet_hls/baseline_100MHz/impl/ip"]
set PRJ_DIR  [file normalize "vivado_prj"]
set OUT_DIR  [file normalize "overlay"]
# Resource limits (shared PC): 5 parallel jobs x 2 threads each = at most
# 10 CPU cores, and the run command caps memory at 10 GB. Without these,
# Vivado starts one job per core, each using several GB of memory.
set JOBS     2
set_param general.maxThreads 2
 
if {![file exists "$HLS_IP/component.xml"]} {
    puts "ERROR: HLS IP not found in $HLS_IP"
    puts "       Add 'export_design -format ip_catalog' to script.tcl and run vitis_hls first."
    exit 1
}
 
# ---------------------------------------------------------------- project
create_project mobilenet_bd $PRJ_DIR -part xck26-sfvc784-2LV-c -force
 
set bp [lindex [lsort [get_board_parts -quiet "*${BOARD}_som*"]] end]
if {$bp eq ""} {
    puts "INFO: $BOARD board files not installed; downloading from the Xilinx board store"
    xhub::refresh_catalog [xhub::get_xstores xilinx_board_store]
    xhub::install [xhub::get_xitems -quiet "*${BOARD}_som*"]
    set_param board.repoPaths [get_property LOCAL_ROOT_DIR [xhub::get_xstores xilinx_board_store]]
    set bp [lindex [lsort [get_board_parts -quiet "*${BOARD}_som*"]] end]
}
if {$bp eq ""} {
    puts "ERROR: no board part found for '$BOARD'. Check BOARD at the top of this script."
    exit 1
}
puts "INFO: using board part $bp"
set_property board_part $bp [current_project]
 
set_property ip_repo_paths $HLS_IP [current_project]
update_ip_catalog
 
set ipdef [lindex [get_ipdefs -filter {NAME == mobilenet_top}] 0]
if {$ipdef eq ""} {
    puts "ERROR: mobilenet_top IP not in the catalog (check $HLS_IP)"
    exit 1
}
 
# ----------------------------------------------------------- block design
create_bd_design design_1
 
# Processor with the board's presets (DDR, clocks, MIO)
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} $ps
 
# One master port for control, one high-performance slave port into DDR,
# and the PL clock frequency
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $CLK_MHZ \
] $ps
 
# The accelerator
set acc [create_bd_cell -type ip -vlnv $ipdef mobilenet_top_0]
 
# Control path: PS -> accelerator registers (start, buffer addresses)
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} \
    Slave {/mobilenet_top_0/s_axi_ctrl} \
    ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}] \
    [get_bd_intf_pins mobilenet_top_0/s_axi_ctrl]
 
# Data path: the accelerator's 4 memory ports -> one SmartConnect -> PS HP0
# port -> DDR. Wired explicitly (v1 of this script used apply_bd_automation
# here, which Vivado refused).
set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc_mem]
set_property -dict [list CONFIG.NUM_SI {4} CONFIG.NUM_MI {1}] $smc
foreach i {0 1 2 3} {
    connect_bd_intf_net [get_bd_intf_pins mobilenet_top_0/m_axi_gmem$i] \
                        [get_bd_intf_pins smc_mem/S0${i}_AXI]
}
connect_bd_intf_net [get_bd_intf_pins smc_mem/M00_AXI] \
                    [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
 
# Same 100 MHz clock (pl_clk0) for the SmartConnect and the HP0 port. The
# control-path automation above already put the accelerator on pl_clk0.
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
               [get_bd_pins smc_mem/aclk] \
               [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk]
 
# Reset: reuse the reset block the control-path automation created
set rst [lindex [get_bd_cells -filter {VLNV =~ "*proc_sys_reset*"}] 0]
if {$rst eq ""} {
    set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_pl0]
    connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins $rst/slowest_sync_clk]
    connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins $rst/ext_reset_in]
}
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins smc_mem/aresetn]
 
# Done interrupt (optional: the host program polls the status register)
if {[llength [get_bd_pins -quiet zynq_ultra_ps_e_0/pl_ps_irq0]] > 0} {
    connect_bd_net [get_bd_pins mobilenet_top_0/interrupt] \
                   [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]
}
 
assign_bd_address
validate_bd_design
save_bd_design
 
# ------------------------------------------------- synthesize + implement
make_wrapper -files [get_files design_1.bd] -top -import
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1
 
launch_runs impl_1 -to_step write_bitstream -jobs $JOBS
wait_on_run impl_1
 
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    puts "ERROR: implementation did not finish. See $PRJ_DIR/mobilenet_bd.runs/impl_1/runme.log"
    exit 1
}
 
# ------------------------------------------------------------ outputs
file mkdir $OUT_DIR
set bit [lindex [glob -nocomplain $PRJ_DIR/mobilenet_bd.runs/impl_1/*.bit] 0]
set hwh [lindex [glob -nocomplain $PRJ_DIR/mobilenet_bd.gen/sources_1/bd/design_1/hw_handoff/*.hwh] 0]
if {$bit eq "" || $hwh eq ""} {
    puts "ERROR: bitstream or .hwh not found (bit='$bit' hwh='$hwh')"
    exit 1
}
file copy -force $bit $OUT_DIR/mobilenet.bit
file copy -force $hwh $OUT_DIR/mobilenet.hwh
 
open_run impl_1
report_utilization    -file $OUT_DIR/utilization.rpt
report_timing_summary -file $OUT_DIR/timing.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "INFO: worst setup slack (WNS) = $wns ns  (>= 0 means timing is met at $CLK_MHZ MHz)"
puts "INFO: done. Copy the folder $OUT_DIR to the board."
