# Vitis HLS `matvec` IP -> PYNQ-Z2 overlay (overlay/matvec.bit + overlay/matvec.hwh), Vivado 2025.2.
#
#   vivado -mode batch -source vivado/build_overlay.tcl -tclargs <ip_repo_dir> [--jobs N] [--bd-only]
#   vivado -mode batch -source vivado/build_overlay.tcl -tclargs --no-ip
#
# <ip_repo_dir> : folder with the exported HLS IP (the one containing component.xml, normally
#                 <hls project>/<solution>/impl/ip). Must provide xilinx.com:hls:matvec.
# --jobs N      : parallel synthesis/implementation jobs (default 4)
# --bd-only     : build + validate + generate the wrapper, stop before synthesis (nothing copied to overlay/)
# --no-ip       : PS7 + reset only, validate and stop (checks the Vivado setup without the IP); implies --bd-only
#
# Design (everything on FCLK_CLK0):
#   PS7 M_AXI_GP0 -> smartconnect -> matvec_0/s_axi_control   @ 0x43C00000, 64K
#   matvec_0/m_axi_gmem0 -> smartconnect -> PS7 S_AXI_HP0     DDR 512M @ 0x0
#   matvec_0/m_axi_gmem1 -> smartconnect -> PS7 S_AXI_HP1     DDR 512M @ 0x0
# The interface names above are the ones Vitis HLS generates for the bundles in hls/matvec.h; if the kernel's
# bundles are renamed, the check right after create_bd_cell fails with the list of pins that do exist.

set part     xc7z020clg400-1
set bd_name  matvec
set ctrl_base 0x43C00000
set ctrl_range 64K
set ddr_range  512M

set script_dir [file dirname [file normalize [info script]]]
set root       [file dirname $script_dir]
set proj_dir   [file join $script_dir build]
set out_dir    [file join $root overlay]

# ---- arguments ----
set jobs 4
set bd_only 0
set no_ip 0
set ip_repo ""
for {set i 0} {$i < [llength $argv]} {incr i} {
    set a [lindex $argv $i]
    switch -- $a {
        --jobs    { incr i; set jobs [lindex $argv $i] }
        --bd-only { set bd_only 1 }
        --no-ip   { set no_ip 1; set bd_only 1 }
        default   { set ip_repo [file normalize $a] }
    }
}
if {![string is integer -strict $jobs] || $jobs < 1} { error "--jobs needs a positive integer" }
if {!$no_ip && ![file isdirectory $ip_repo]} {
    error "usage: vivado -mode batch -source vivado/build_overlay.tcl -tclargs <ip_repo_dir> \[--jobs N\] \[--bd-only\]  (or --no-ip)"
}

# ---- project ----
create_project -force $bd_name $proj_dir -part $part
if {!$no_ip} {
    set_property ip_repo_paths $ip_repo [current_project]
    update_ip_catalog -rebuild
    set kernel_vlnv [lindex [lsort -dictionary [get_ipdefs -all xilinx.com:hls:matvec:*]] end]
    if {$kernel_vlnv eq ""} { error "no xilinx.com:hls:matvec IP found under $ip_repo (point at the folder containing component.xml)" }
    puts "INFO: kernel IP $kernel_vlnv from $ip_repo"
}

# ---- block design ----
create_bd_design $bd_name

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 processing_system7_0]
# PS7 defaults are already DDR3 / 512 MB (HP DDR segment = 0x0..0x1FFFFFFF), which is what the PYNQ-Z2 has.
# FCLK0 is requested at 150 MHz (the IP derives it as 1200 MHz IO PLL / 4 / 2; Vivado times the design at 150).
# On the board PYNQ re-programs the FCLK divisors from the PLLs the boot configured, so 142.857 MHz is likely; slower is safe.
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 1 \
    CONFIG.PCW_USE_S_AXI_HP0 1 CONFIG.PCW_S_AXI_HP0_DATA_WIDTH 64 \
    CONFIG.PCW_USE_S_AXI_HP1 1 CONFIG.PCW_S_AXI_HP1_DATA_WIDTH 64 \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 150] $ps
make_bd_intf_pins_external [get_bd_intf_pins $ps/DDR]
make_bd_intf_pins_external [get_bd_intf_pins $ps/FIXED_IO]

set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset proc_sys_reset_0]
connect_bd_net [get_bd_pins $ps/FCLK_CLK0]     [get_bd_pins $rst/slowest_sync_clk]
connect_bd_net [get_bd_pins $ps/FCLK_RESET0_N] [get_bd_pins $rst/ext_reset_in]
set clk_pins [list $ps/M_AXI_GP0_ACLK $ps/S_AXI_HP0_ACLK $ps/S_AXI_HP1_ACLK]
set rst_pins {}

if {!$no_ip} {
    set k [create_bd_cell -type ip -vlnv $kernel_vlnv matvec_0]
    foreach p {s_axi_control m_axi_gmem0 m_axi_gmem1} {
        if {[get_bd_intf_pins -quiet $k/$p] eq ""} {
            error "matvec_0 has no interface '$p'; it has: [get_bd_intf_pins -of_objects $k]"
        }
    }
    lappend clk_pins $k/ap_clk
    lappend rst_pins $k/ap_rst_n

    # HLS m_axi is AXI4, PS7 HP/GP are AXI3: SmartConnect does the protocol (and burst length) conversion.
    foreach {smc  si_pin           mi_pin} [list \
        smc_ctrl $ps/M_AXI_GP0     $k/s_axi_control \
        smc_hp0  $k/m_axi_gmem0    $ps/S_AXI_HP0 \
        smc_hp1  $k/m_axi_gmem1    $ps/S_AXI_HP1] {
        set c [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect $smc]
        set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1] $c
        connect_bd_intf_net [get_bd_intf_pins $si_pin] [get_bd_intf_pins $c/S00_AXI]
        connect_bd_intf_net [get_bd_intf_pins $c/M00_AXI] [get_bd_intf_pins $mi_pin]
        lappend clk_pins $c/aclk
        lappend rst_pins $c/aresetn
    }
}

foreach p $clk_pins { connect_bd_net [get_bd_pins $ps/FCLK_CLK0] [get_bd_pins $p] }
foreach p $rst_pins { connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $p] }

if {!$no_ip} {
    assign_bd_address -target_address_space [get_bd_addr_spaces $ps/Data] \
        -offset $ctrl_base -range $ctrl_range [get_bd_addr_segs $k/s_axi_control/Reg]
    foreach g {gmem0 gmem1} hp {HP0 HP1} {
        assign_bd_address -target_address_space [get_bd_addr_spaces $k/Data_m_axi_$g] \
            -offset 0x0 -range $ddr_range [get_bd_addr_segs $ps/S_AXI_$hp/${hp}_DDR_LOWOCM]
    }
    puts "INFO: address map:"
    foreach as [get_bd_addr_spaces] {
        foreach s [get_bd_addr_segs -of_objects $as] {
            puts "  $s  offset [get_property OFFSET $s]  range [get_property RANGE $s]"
        }
    }
}

validate_bd_design
save_bd_design
puts "INFO: FCLK_CLK0 = [expr {[get_property CONFIG.FREQ_HZ [get_bd_pins $ps/FCLK_CLK0]] / 1e6}] MHz"

set bd_file [get_files $bd_name.bd]
make_wrapper -files $bd_file -top -import
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1
generate_target all $bd_file

if {$bd_only} {
    puts "INFO: --bd-only: block design validated and wrapper generated in $proj_dir"
    return
}

# ---- synthesis, implementation, bitstream ----
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
foreach r {synth_1 impl_1} {
    if {[get_property PROGRESS [get_runs $r]] ne "100%"} { error "$r did not complete: [get_property STATUS [get_runs $r]]" }
}

set impl_dir [get_property DIRECTORY [get_runs impl_1]]
set bit [file join $impl_dir ${bd_name}_wrapper.bit]
set hwh [file join $proj_dir $bd_name.gen sources_1 bd $bd_name hw_handoff $bd_name.hwh]
if {![file exists $bit]} { error "bitstream not found: $bit" }
if {![file exists $hwh]} { error "hardware handoff not found: $hwh" }

file mkdir $out_dir
file copy -force $bit [file join $out_dir $bd_name.bit]
file copy -force $hwh [file join $out_dir $bd_name.hwh]

set wns [get_property STATS.WNS [get_runs impl_1]]
puts "INFO: copied $bit -> [file join $out_dir $bd_name.bit]"
puts "INFO: copied $hwh -> [file join $out_dir $bd_name.hwh]"
puts "RESULT: implemented WNS = $wns ns  (TNS [get_property STATS.TNS [get_runs impl_1]] ns, WHS [get_property STATS.WHS [get_runs impl_1]] ns)"
if {[string is double -strict $wns] && $wns < 0} { puts "WARNING: timing not met at [expr {[get_property CONFIG.FREQ_HZ [get_bd_pins $ps/FCLK_CLK0]] / 1e6}] MHz; lower PCW_FPGA0_PERIPHERAL_FREQMHZ or fix the kernel's critical path" }
