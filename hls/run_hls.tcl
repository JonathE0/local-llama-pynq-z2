# C-sim, C-synthesis and Vivado IP export for hls/matvec.cpp.
# Run from anywhere:  vitis-run --mode hls --tcl hls/run_hls.tcl
# Optional env vars: COSIM=1 runs RTL co-simulation on the small shapes; IMPL=1 also runs Vivado
# place & route on the exported IP to check timing (report in .../sol/impl/report).
# Output: hls/build/matvec_prj/sol/  (IP repository: .../sol/impl/ip); refreshes hls/xmatvec_hw.h
set here [file dirname [file normalize [info script]]]
file mkdir $here/build
cd $here/build

open_project -reset matvec_prj
set_top matvec
add_files $here/matvec.cpp
add_files -tb $here/matvec_tb.cpp -cflags "-I$here"
open_solution -reset sol -flow_target vivado
set_part xc7z020clg400-1
create_clock -period 6.667 -name default
config_interface -m_axi_addr64=false
# matvec.h guarantees 64-byte aligned buffers; lets HLS widen the float accesses on the 64-bit gmem1 into bursts
config_interface -m_axi_alignment_byte_size=64

csim_design
csynth_design
if {[info exists ::env(COSIM)]} {
    cosim_design -argv quick
}
if {[info exists ::env(IMPL)]} {
    export_design -format ip_catalog -flow impl
} else {
    export_design -format ip_catalog
}
# keep the committed register map in sync with the exported IP
file copy -force $here/build/matvec_prj/sol/impl/ip/drivers/matvec_v1_0/src/xmatvec_hw.h $here/xmatvec_hw.h
exit
