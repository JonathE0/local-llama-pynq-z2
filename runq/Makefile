# On the board, with the repo layout kept: <dir>/{driver,hls,runq}. Run `make` in runq/.
#   runq       CPU only (same as upstream runq, minus the float copy of the embedding table)
#   runq-fpga  matmul() on the FPGA kernel; needs hls/xmatvec_hw.h (from Vitis HLS) and XRT
CFLAGS = -O3 -ffast-math -mcpu=cortex-a9 -mfpu=neon -fopenmp
DRV    = ../driver
HLS    = ../hls
XRT    = /usr/include/xrt

all: runq runq-fpga

runq: runq.c
	gcc $(CFLAGS) $< -o $@ -lm

runq-fpga: runq.c $(DRV)/matvec_fpga.cpp $(DRV)/matvec_fpga.h $(HLS)/xmatvec_hw.h
	gcc $(CFLAGS) -DFPGA -I$(DRV) -c runq.c -o runq-fpga.o
	g++ $(CFLAGS) -I$(DRV) -I$(HLS) -I$(XRT) -c $(DRV)/matvec_fpga.cpp -o matvec_fpga.o
	g++ $(CFLAGS) runq-fpga.o matvec_fpga.o -o $@ -lxrt_coreutil -lm

clean:
	rm -f runq runq-fpga *.o

.PHONY: all clean
