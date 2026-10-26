// int8 matrix x vector for llama2.c runq.c matmul(). Interface and math: see matvec.h.
//
// Dataflow pipeline, one 64-bit weight word (8 MACs) per cycle:
//   read_w : wq -> stream, one long sequential burst over all d*n/8 words (gmem0)
//   read_s : ws -> stream, prefetched independently so it never stalls the weights (gmem1)
//   dot    : 8 int8 MACs/cycle, int32 sum per group, p = (float)ival * ws * xs (same op order as runq.c)
//   acc    : float sum of p over the groups of a row, in runq.c order (bit-exact).
//            One add chain per row can't take a new group every 4 cycles (gs=32): the loop-carried fadd
//            is ~10 cycles at 150 MHz. So row r goes to stream p[r % K] and each iteration adds one group
//            of K consecutive rows: K independent chains, each still sequential within its row.
//   write_out : out[], sequential burst (gmem1)
// xq/xs are copied to on-chip memory once per call, before the pipeline starts.
#include "matvec.h"
#include <hls_stream.h>

typedef ap_uint<64> word_t;
#define MAXW (MATVEC_MAX_N / 8)
#define MAXG (MATVEC_MAX_N / 32)
#define K 4 // rows accumulated in parallel by acc(); needs K / (acc II) >= 1/4 group per cycle

// n / gs for gs in {32, 64} without a divider
static int div_gs(int x, int gs) { return gs == 64 ? x >> 6 : x >> 5; }

// loop_tripcount values below are for n=768 d=2048 gs=64 and only affect the latency report.

static void read_w(const word_t *wq, hls::stream<word_t> &wst, int n, int d) {
    const int total = d * (n / 8);
    for (int t = 0; t < total; t++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=196608 max=196608
        wst.write(wq[t]);
    }
}

// Floats are moved in pairs: gmem1 is 64 bits wide (xq), and HLS only bursts a float array on it when it can
// widen two consecutive floats into one 64-bit beat. A lone float at an odd end is a single access.
static void read_s(const float *ws, hls::stream<float> &sst, int n, int d, int gs) {
    const int total = d * div_gs(n, gs);
    for (int t = 0; t < total / 2; t++) {
#pragma HLS pipeline II=2
#pragma HLS loop_tripcount min=12288 max=12288
        float a = ws[2 * t], b = ws[2 * t + 1];
        sst.write(a);
        sst.write(b);
    }
    if (total & 1) sst.write(ws[total - 1]);
}

static void dot(hls::stream<word_t> &wst, hls::stream<float> &sst, const word_t xq[MAXW], const float xs[MAXG],
                hls::stream<float> p[K], int n, int d, int gs) {
    const int nw = n / 8, gw = gs / 8, total = d * nw;
    int j = 0, k = 0, g = 0; // word in row, word in group, group in row
    ap_int<22> ival = 0;     // group sum: |ival| <= 64 * 128 * 128 = 2^20
    ap_uint<K> sel = 1;      // one-hot row % K (an int compare here was the post-route critical path)
    for (int t = 0; t < total; t++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=196608 max=196608
        word_t w = wst.read(), x = xq[j];
        ap_int<19> sum = 0;
        for (int e = 0; e < 8; e++) {
#pragma HLS unroll
            ap_int<8> a = w.range(8 * e + 7, 8 * e), b = x.range(8 * e + 7, 8 * e);
            sum += a * b;
        }
        ival = (k == 0 ? ap_int<22>(0) : ival) + sum;
        const bool eog = k == gw - 1, eor = j == nw - 1;
        if (eog) {
            float s = sst.read();
            float prod = ((float)ival.to_int()) * s * xs[g];
            for (int c = 0; c < K; c++) {
#pragma HLS unroll
                if (sel[c]) p[c].write(prod);
            }
        }
        k = eog ? 0 : k + 1;
        g = eor ? 0 : (eog ? g + 1 : g);
        j = eor ? 0 : j + 1;
        if (eor) sel = (sel << 1) | (sel >> (K - 1));
    }
}

static void acc(hls::stream<float> p[K], hls::stream<float> &ost, int n, int d, int gs) {
    const int ng = div_gs(n, gs), total = ((d + K - 1) / K) * ng;
    int g = 0, r = 0; // group in row, first row of the current block of K rows
    float v[K] = {};
#pragma HLS array_partition variable=v complete
    for (int t = 0; t < total; t++) {
        // II = loop-carried fadd (9 stages at 150 MHz) + select. K rows per 10 cycles = 2.5 cycles per group,
        // under the 4 (gs=32) or 8 (gs=64) cycles dot needs per group.
#pragma HLS pipeline II=10
#pragma HLS loop_tripcount min=6144 max=6144
        const bool last = g == ng - 1;
        for (int c = 0; c < K; c++) {
#pragma HLS unroll
            if (r + c < d) v[c] = (g == 0 ? 0.0f : v[c]) + p[c].read(); // runq.c: val = 0.0f; val += ...
        }
        if (last) {
            for (int c = 0; c < K; c++) {
#pragma HLS unroll
                if (r + c < d) ost.write(v[c]);
            }
        }
        g = last ? 0 : g + 1;
        r = last ? r + K : r;
    }
}

static void write_out(hls::stream<float> &ost, float *out, int d) {
    for (int i = 0; i < d / 2; i++) {
#pragma HLS pipeline II=2
#pragma HLS loop_tripcount min=1024 max=1024
        float a = ost.read(), b = ost.read();
        out[2 * i] = a;
        out[2 * i + 1] = b;
    }
    if (d & 1) out[d - 1] = ost.read();
}

static void run(const word_t *wq, const float *ws, float *out, const word_t xq[MAXW], const float xs[MAXG],
                int n, int d, int gs) {
#pragma HLS dataflow
    hls::stream<word_t, 64> wst;
    hls::stream<float, 64> sst;
    // dot writes rows 0..K-2 of a block completely before acc can start it (acc waits for row K-1), so each
    // p[c] must hold n/gs <= MAXG products or the pipeline deadlocks; 2*MAXG also absorbs acc's lag.
    hls::stream<float, 2 * MAXG> p[K];
    hls::stream<float, 16> ost;
    read_w(wq, wst, n, d);
    read_s(ws, sst, n, d, gs);
    dot(wst, sst, xq, xs, p, n, d, gs);
    acc(p, ost, n, d, gs);
    write_out(ost, out, d);
}

void matvec(const ap_uint<64> *wq, const float *ws, const ap_uint<64> *xq, const float *xs, float *out,
            int n, int d, int gs) {
    // depth= only sizes the cosim memory model (largest "quick" testbench shape: n=768 d=768 gs=64)
#pragma HLS interface m_axi port=wq bundle=gmem0 offset=slave max_read_burst_length=256 num_read_outstanding=8 max_widen_bitwidth=64 depth=73728
#pragma HLS interface m_axi port=ws bundle=gmem1 offset=slave max_widen_bitwidth=64 depth=9216
#pragma HLS interface m_axi port=xq bundle=gmem1 offset=slave max_widen_bitwidth=64 depth=96
#pragma HLS interface m_axi port=xs bundle=gmem1 offset=slave max_widen_bitwidth=64 depth=12
#pragma HLS interface m_axi port=out bundle=gmem1 offset=slave max_widen_bitwidth=64 depth=768
#pragma HLS interface s_axilite port=wq bundle=control
#pragma HLS interface s_axilite port=ws bundle=control
#pragma HLS interface s_axilite port=xq bundle=control
#pragma HLS interface s_axilite port=xs bundle=control
#pragma HLS interface s_axilite port=out bundle=control
#pragma HLS interface s_axilite port=n bundle=control
#pragma HLS interface s_axilite port=d bundle=control
#pragma HLS interface s_axilite port=gs bundle=control
#pragma HLS interface s_axilite port=return bundle=control
    word_t xq_buf[MAXW];
    float xs_buf[MAXG];
load_xq:
    for (int i = 0; i < n / 8; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=96 max=96
        xq_buf[i] = xq[i];
    }
    const int ng = div_gs(n, gs);
load_xs:
    for (int i = 0; i < ng / 2; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=6 max=6
        xs_buf[2 * i] = xs[2 * i];
        xs_buf[2 * i + 1] = xs[2 * i + 1];
    }
    if (ng & 1) xs_buf[ng - 1] = xs[ng - 1];
    run(wq, ws, out, xq_buf, xs_buf, n, d, gs);
}
