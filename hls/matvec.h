// Contract between the HLS kernel (hls/matvec.cpp) and the board driver (driver/).
// Computes what matmul() in llama2.c runq.c computes, for one weight matrix W (d rows, n columns):
//   out[i] = sum over groups g of ( sum_{k<gs} wq[i*n + g*gs + k] * xq[g*gs + k] ) * ws[(i*n)/gs + g] * xs[g]
// with int32 accumulation inside a group and float accumulation across groups.
#pragma once
#include <ap_int.h>

#define MATVEC_MAX_N 8192 // longest input vector (Qwen2.5-0.5B FFN is 4864)

// wq  : d*n int8 weights, row-major, packed 8 per 64-bit word, element k of a word in bits [8k+7:8k]
// ws  : d*n/gs float weight scales
// xq  : n int8 activations, packed like wq
// xs  : n/gs float activation scales
// out : d floats
// Requires n % gs == 0, gs in {32, 64}, n <= MATVEC_MAX_N. Buffers are physically contiguous, 64-byte aligned.
// AXI: wq on m_axi bundle gmem0 (-> S_AXI_HP0); ws, xq, xs, out on m_axi bundle gmem1 (-> S_AXI_HP1);
//      all arguments and return on s_axilite bundle control.
void matvec(const ap_uint<64> *wq, const float *ws, const ap_uint<64> *xq, const float *xs, float *out,
            int n, int d, int gs);
