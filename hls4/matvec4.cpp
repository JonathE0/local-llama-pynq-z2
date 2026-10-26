// Qwen int4/int8 matrix x vector. Interface and math: see matvec4.h. Same structure as hls/matvec.cpp.
//
// Dataflow pipeline, one 64-bit weight word per cycle (16 int4 or 8 int8 MACs):
//   read_w : w -> stream, one long sequential burst over all weight words (gmem0)
//   read_s : ws -> stream, 4 fp16 scales per word, prefetched independently so it never stalls the weights (gmem1)
//   dot    : int sum per group of 32 (2 words at int4, 4 at int8), p = (float)ival * ws * xs
//   acc    : float sum of the p of a row. At int4 a new p comes every 2 cycles but a loop-carried fadd takes ~10,
//            so each iteration takes the next 8 p of a row, adds them as a tree (no carried dependency) and adds
//            that to the row sum: one carried fadd per 8 groups. A row's last batch is padded with zeros.
//   write_out : out[], sequential burst (gmem1)
// xq/xs are copied to on-chip memory once per call, before the pipeline starts.
#include "matvec4.h"
#include <hls_stream.h>

typedef ap_uint<64> word_t;
#define MAXX (MATVEC4_MAX_N / 16) // xq words per half (even words in xlo, odd in xhi)
#define MAXG (MATVEC4_MAX_N / 32)

// loop_tripcount values below are for n=896 d=4864 wbits=4 and only affect the latency report.

// fp16 bits -> float, exact. Done by hand: the Xilinx floating-point cores flush subnormal inputs to zero.
static float h2f(ap_uint<16> h) {
    const ap_uint<5> e = h(14, 10);
    const ap_uint<10> m = h(9, 0);
    ap_uint<8> fe = e + 112; // normal: rebias 15 -> 127
    ap_uint<10> fm = m;
    if (e == 31) {
        fe = 255; // inf / nan
    } else if (e == 0) {
        if (m == 0) {
            fe = 0;
        } else { // subnormal m * 2^-24: move the leading one out of the mantissa
            const int lz = m.countLeadingZeros();
            fe = 112 - lz;
            fm = m << (lz + 1);
        }
    }
    union { unsigned int u; float f; } r;
    r.u = ((unsigned int)h[15] << 31) | ((unsigned int)fe << 23) | ((unsigned int)fm << 13);
    return r.f;
}

// total = d * nw weight words, computed once in matvec4() (each `d * x` costs 3 DSPs)
static void read_w(const word_t *w, hls::stream<word_t> &wst, int total) {
    for (int t = 0; t < total; t++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=272384 max=272384
        wst.write(w[t]);
    }
}

static void read_s(const word_t *ws, hls::stream<word_t> &sst, int total, bool w4) {
    const int words = ((w4 ? total >> 1 : total >> 2) + 3) >> 2; // d * n/32 fp16, 4 per word
    for (int t = 0; t < words; t++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=34048 max=34048
        sst.write(ws[t]);
    }
}

static void dot(hls::stream<word_t> &wst, hls::stream<word_t> &sst, const word_t xlo[MAXX], const word_t xhi[MAXX],
                const float xs[MAXG], hls::stream<float> &pst, int nw, int total, bool w4) {
    const int gw = w4 ? 2 : 4;
    int j = 0, k = 0, g = 0; // word in row, word in group, group in row
    ap_uint<2> q = 0;        // fp16 scale in the current ws word (scales run on across rows)
    word_t sw = 0;
    ap_int<22> ival = 0;     // group sum: |ival| <= 32 * 128 * 128 = 2^19
    for (int t = 0; t < total; t++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=272384 max=272384
        const word_t w = wst.read();
        const int a = w4 ? j : j >> 1; // int4 word j covers xq words 2j, 2j+1; int8 word j covers xq word j
        const word_t lo = xlo[a], hi = xhi[a];
        const ap_uint<128> x16 = (hi, lo);
        ap_int<16> s4 = 0; // |s4| <= 16 * 8 * 128 = 2^14
        for (int e = 0; e < 16; e++) {
#pragma HLS unroll
            const ap_int<4> wv = w.range(4 * e + 3, 4 * e);
            const ap_int<8> xv = x16.range(8 * e + 7, 8 * e);
            const ap_int<12> m = wv * xv;
#pragma HLS bind_op variable=m op=mul impl=fabric
            s4 += m;
        }
        const word_t x8 = (j & 1) ? hi : lo;
        ap_int<19> s8 = 0; // |s8| <= 8 * 128 * 128 = 2^17
        for (int e = 0; e < 8; e++) {
#pragma HLS unroll
            const ap_int<8> wv = w.range(8 * e + 7, 8 * e), xv = x8.range(8 * e + 7, 8 * e);
            const ap_int<16> m = wv * xv; // LUTs too: keeps the DSP count down to the float units
#pragma HLS bind_op variable=m op=mul impl=fabric
            s8 += m;
        }
        ival = (k == 0 ? ap_int<22>(0) : ival) + (w4 ? ap_int<19>(s4) : s8);
        const bool eog = k == gw - 1, eor = j == nw - 1;
        if (eog) {
            if (q == 0) sw = sst.read();
            const float s = h2f(sw.range(16 * q + 15, 16 * q));
            pst.write((float)ival.to_int() * s * xs[g]);
            q++;
        }
        k = eog ? 0 : k + 1;
        g = eor ? 0 : (eog ? g + 1 : g);
        j = eor ? 0 : j + 1;
    }
}

static void acc(hls::stream<float> &pst, hls::stream<float> &ost, int n, int d) {
    const int ng = n / 32, nb = (ng + 7) / 8, tail = ng - 8 * (nb - 1);
    int b = 0, r = 0; // batch of 8 groups in row, row
    float v = 0;
    while (r < d) { // d * nb iterations, counted by rows to save a multiplier
#pragma HLS pipeline II=10
        // II: carried fadd (9 stages at 150 MHz) + select. acc keeps up while 10 * ceil(n/256) <= n/16 cycles per row
        // (int4; n=896: 40 <= 56, n=4864: 190 <= 304). ponytail: n < 160 or n = 288 run acc-bound; fine for Qwen.
#pragma HLS loop_tripcount min=19456 max=19456
        const bool last = b == nb - 1;
        float p[8];
        for (int c = 0; c < 8; c++) {
#pragma HLS unroll
            p[c] = (!last || c < tail) ? pst.read() : 0.0f;
        }
        const float s = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
        v = (b == 0 ? 0.0f : v) + s;
        if (last) {
            ost.write(v);
            r++;
        }
        b = last ? 0 : b + 1;
    }
}

// Floats are moved in pairs: gmem1 is 64 bits wide, and HLS only bursts a float array on it when it can
// widen two consecutive floats into one 64-bit beat. A lone float at an odd end is a single access.
static void write_out(hls::stream<float> &ost, float *out, int d) {
    for (int i = 0; i < d / 2; i++) {
#pragma HLS pipeline II=2
#pragma HLS loop_tripcount min=2432 max=2432
        float a = ost.read(), b = ost.read();
        out[2 * i] = a;
        out[2 * i + 1] = b;
    }
    if (d & 1) out[d - 1] = ost.read();
}

static void run(const word_t *w, const word_t *ws, float *out, const word_t xlo[MAXX], const word_t xhi[MAXX],
                const float xs[MAXG], int n, int d, int nw, int total, bool w4) {
#pragma HLS dataflow
    hls::stream<word_t, 64> wst;
    hls::stream<word_t, 16> sst;
    hls::stream<float, 32> pst;
    hls::stream<float, 16> ost;
    read_w(w, wst, total);
    read_s(ws, sst, total, w4);
    dot(wst, sst, xlo, xhi, xs, pst, nw, total, w4);
    acc(pst, ost, n, d);
    write_out(ost, out, d);
}

void matvec4(const ap_uint<64> *w, const ap_uint<64> *ws, const ap_uint<64> *xq, const float *xs, float *out,
             int n, int d, int wbits) {
    // depth= only sizes the cosim memory model (largest testbench cosim shapes: n=896 d=4864 wbits=4, n=8192)
#pragma HLS interface m_axi port=w bundle=gmem0 offset=slave max_read_burst_length=256 num_read_outstanding=8 max_widen_bitwidth=64 depth=272384
#pragma HLS interface m_axi port=ws bundle=gmem1 offset=slave max_widen_bitwidth=64 depth=34048
#pragma HLS interface m_axi port=xq bundle=gmem1 offset=slave max_widen_bitwidth=64 depth=1024
#pragma HLS interface m_axi port=xs bundle=gmem1 offset=slave max_widen_bitwidth=64 depth=256
#pragma HLS interface m_axi port=out bundle=gmem1 offset=slave max_widen_bitwidth=64 depth=4864
#pragma HLS interface s_axilite port=w bundle=control
#pragma HLS interface s_axilite port=ws bundle=control
#pragma HLS interface s_axilite port=xq bundle=control
#pragma HLS interface s_axilite port=xs bundle=control
#pragma HLS interface s_axilite port=out bundle=control
#pragma HLS interface s_axilite port=n bundle=control
#pragma HLS interface s_axilite port=d bundle=control
#pragma HLS interface s_axilite port=wbits bundle=control
#pragma HLS interface s_axilite port=return bundle=control
    word_t xlo[MAXX], xhi[MAXX];
    float xs_buf[MAXG];
load_xq:
    for (int i = 0; i < n / 8; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=112 max=112
        const word_t v = xq[i];
        if (i & 1) xhi[i >> 1] = v;
        else xlo[i >> 1] = v;
    }
    const int ng = n / 32;
load_xs:
    for (int i = 0; i < ng / 2; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=14 max=14
        xs_buf[2 * i] = xs[2 * i];
        xs_buf[2 * i + 1] = xs[2 * i + 1];
    }
    if (ng & 1) xs_buf[ng - 1] = xs[ng - 1];
    const bool w4 = wbits == 4;
    const int nw = w4 ? n / 16 : n / 8; // weight words per row
    run(w, ws, out, xlo, xhi, xs_buf, n, d, nw, d * nw, w4);
}
