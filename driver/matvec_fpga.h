// Board driver for the HLS matvec kernels (contracts: hls/matvec.h; hls4/matvec4.h when matvec_fpga.cpp is built
// with -DMATVEC4 -I../hls4). C API on top of XRT + /dev/mem.
// The overlay must already be loaded (e.g. with pynq). Needs root for /dev/mem and /dev/dri.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Open the XRT device and map the kernel's control registers (0x43C00000, 64 KB).
// Env FPGA_MEMIDX = memory bank index for buffers (default 0).
// Returns 0, or -1 with a message on stderr.
int fpga_open(void);

// Allocate a physically contiguous, non-cacheable buffer (page aligned, kept until fpga_close()).
// Returns the CPU pointer and stores the physical address in *phys; NULL with a message on stderr on failure.
void *fpga_alloc(size_t bytes, uint64_t *phys);

// Same, but silent on failure (for callers that fall back to a smaller size).
void *fpga_alloc_try(size_t bytes, uint64_t *phys);

// Run one matvec (semantics in hls/matvec.h) and wait for it to finish. Arguments are physical addresses.
// Returns 0, or -1 with a message on stderr (bad arguments, or the kernel did not finish).
int fpga_matvec(uint64_t wq, uint64_t ws, uint64_t xq, uint64_t xs, uint64_t out, int n, int d, int gs);

// -DMATVEC4 build only (instead of fpga_matvec): one matvec4 (semantics in hls4/matvec4.h), waits for it.
// Physical addresses, each 64-byte aligned; wbits 4 or 8. Returns 0, or -1 with a message on stderr.
int fpga_matvec4(uint64_t w, uint64_t ws, uint64_t xq, uint64_t xs, uint64_t out, int n, int d, int wbits);

// Unmap the registers and free all buffers.
void fpga_close(void);

#ifdef __cplusplus
}
#endif
