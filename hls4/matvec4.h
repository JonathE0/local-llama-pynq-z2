// Contract between the Qwen kernel (hls4/matvec4.cpp) and the board driver (driver/).
// One matrix-vector product with group size 32, the layout of tools/gguf2qwb.py's .qwb matrices:
//   out[i] = sum over groups g of ( sum_{k<32} w[i*n + g*32 + k] * xq[g*32 + k] ) * ws[(i*n)/32 + g] * xs[g]
// int32 accumulation inside a group; float accumulation across groups (any order: checked with a tolerance).
#pragma once
#include <ap_int.h>

#define MATVEC4_MAX_N 8192 // longest input vector (Qwen2.5-0.5B FFN down projection is 4864)

// w     : d*n weights, row-major, packed little-endian into 64-bit words.
//         wbits=4: two's complement int4, element 2i of the matrix in the low nibble of byte i, 2i+1 in the high one
//         wbits=8: int8
// ws    : d*n/32 IEEE fp16 weight scales, 4 per 64-bit word, element j of a word in bits [16j+15:16j]
// xq    : n int8 activations, 8 per 64-bit word (element k in bits [8k+7:8k])
// xs    : n/32 float32 activation scales
// out   : d float32 results
// Requires n % 32 == 0, n <= MATVEC4_MAX_N, d >= 1, wbits in {4, 8}; buffers physically contiguous, 64-byte aligned.
// AXI: w on m_axi bundle gmem0 (-> S_AXI_HP0); ws, xq, xs, out on m_axi bundle gmem1 (-> S_AXI_HP1);
//      all arguments and return on s_axilite bundle control (registers at 0x43C00000 in the qwen overlay).
void matvec4(const ap_uint<64> *w, const ap_uint<64> *ws, const ap_uint<64> *xq, const float *xs, float *out,
             int n, int d, int wbits);
