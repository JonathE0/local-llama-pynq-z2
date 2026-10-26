# On the board, with the repo layout kept: <dir>/{driver,hls4,qwen}. Run `make` in qwen/.
#   qwen       CPU only
#   qwen-fpga  every matmul on the hls4/ matvec4 kernel; needs hls4/xmatvec4_hw.h (from Vitis HLS) and XRT
CFLAGS = -O3 -ffast-math -mcpu=cortex-a9 -mfpu=neon-fp16 -mfp16-format=ieee -fopenmp
DRV    = ../driver
HLS4   = ../hls4
XRT    = /usr/include/xrt

all: qwen qwen-fpga

qwen: qwen.c tokenizer.c tokenizer.h lcd.c lcd.h
	gcc $(CFLAGS) qwen.c tokenizer.c lcd.c -o $@ -lm

qwen-fpga: qwen.c tokenizer.c tokenizer.h lcd.c lcd.h $(DRV)/matvec_fpga.cpp $(DRV)/matvec_fpga.h $(HLS4)/xmatvec4_hw.h
	gcc $(CFLAGS) -DFPGA -I$(DRV) -c qwen.c -o qwen-fpga.o
	gcc $(CFLAGS) -c tokenizer.c -o tokenizer.o
	gcc $(CFLAGS) -c lcd.c -o lcd.o
	g++ $(CFLAGS) -DMATVEC4 -I$(DRV) -I$(HLS4) -I$(XRT) -c $(DRV)/matvec_fpga.cpp -o matvec4_fpga.o
	g++ $(CFLAGS) qwen-fpga.o tokenizer.o lcd.o matvec4_fpga.o -o $@ -lxrt_coreutil -lm

clean:
	rm -f qwen qwen-fpga *.o

.PHONY: all clean
