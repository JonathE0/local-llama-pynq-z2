# C-sim, C-synthesis and Vivado IP export for hls4/matvec4.cpp.
# Run from anywhere:  vitis-run --mode hls --tcl hls4/run_hls.tcl --work_dir hls4/build
# Optional env vars: COSIM=1 runs RTL co-simulation on the small shapes, COSIM=lat on the two latency shapes
# (n=896 d=4864 int4, n=896 d=2432 int8); IMPL=1 also runs Vivado place & route on the exported IP to check timing
# (report in .../sol/impl/report). csim also checks real matrices from
# models/qwen/qwen2.5-coder-0.5b-q4-head49k.qwb (or $QWB) when present.
# Output: hls4/build/matvec4_prj/sol/  (IP repository: .../sol/impl/ip); refreshes hls4/xmatvec4_hw.h
set here [file dirname [file normalize [info script]]]
file mkdir $here/build
cd $here/build

# real matrices for the testbench (csim and cosim inherit the environment)
if {![info exists ::env(QWB)]} { set ::env(QWB) [file dirname $here]/models/qwen/qwen2.5-coder-0.5b-q4-head49k.qwb }

open_project -reset matvec4_prj
set_top matvec4
add_files $here/matvec4.cpp
add_files -tb $here/matvec4_tb.cpp -cflags "-I$here"
open_solution -reset sol -flow_target vivado
set_part xc7z020clg400-1
create_clock -period 6.667 -name default
config_interface -m_axi_addr64=false
# matvec4.h guarantees 64-byte aligned buffers; lets HLS widen the float accesses on the 64-bit gmem1 into bursts
config_interface -m_axi_alignment_byte_size=64

csim_design
csynth_design
if {[info exists ::env(COSIM)]} {
    cosim_design -argv [expr {$::env(COSIM) eq "lat" ? "lat" : "quick"}]
}
if {[info exists ::env(IMPL)]} {
    export_design -format ip_catalog -flow impl
} else {
    export_design -format ip_catalog
}
# keep the committed register map in sync with the exported IP
file copy -force $here/build/matvec4_prj/sol/impl/ip/drivers/matvec4_v1_0/src/xmatvec4_hw.h $here/xmatvec4_hw.h
exit
