// C-sim / co-sim testbench for matvec4(): random matrices and real ones from the .qwb weight file, checked against a
// double-precision reference. The kernel adds its floats in its own order, so each row passes when
//   |out - ref| <= TOL * sum_g |term_g|,  term_g = (group int sum) * ws * xs
// (float rounding: 2 per term plus the adds, at most ~(5 + n/256) * 2^-24 relative to that sum).
// Usage: no argument = all shapes + real matrices (csim); "quick" = small random shapes (cosim);
//        "lat" = n=896 d=4864 int4 and n=896 d=2432 int8, both 272384 weight words (cosim latency).
// Real matrices: from the .qwb file named by $QWB (skipped if missing).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "matvec4.h"

static const double TOL = 1e-5;
// cosim copies `depth` elements of every m_axi buffer (pragmas in matvec4.cpp), so pad up to that
static const size_t DEPTH_W = 272384, DEPTH_WS = 34048, DEPTH_XQ = 1024, DEPTH_XS = 256, DEPTH_OUT = 4864;

// A matrix in the .qwb / matvec4.h layout: packed weight bytes and fp16 scale bits.
struct Mat {
    int n, d, bits;
    std::vector<uint8_t> w; // d*n*bits/8 bytes
    std::vector<uint16_t> s; // d*n/32 fp16
};

static double half_ref(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const double v = e ? std::ldexp(1024.0 + m, e - 25) : std::ldexp((double)m, -24);
    return (h >> 15) ? -v : v;
}

static int weight(const Mat &m, size_t idx) { // element idx of the row-major matrix
    if (m.bits == 8) return (int8_t)m.w[idx];
    const int nib = (m.w[idx / 2] >> (idx & 1 ? 4 : 0)) & 15;
    return nib >= 8 ? nib - 16 : nib;
}

static std::vector<ap_uint<64> > words(const void *p, size_t bytes, size_t depth) {
    std::vector<ap_uint<64> > v(std::max((bytes + 7) / 8, depth));
    for (size_t i = 0; i < (bytes + 7) / 8; i++) {
        uint64_t x = 0;
        memcpy(&x, (const uint8_t *)p + 8 * i, std::min<size_t>(8, bytes - 8 * i));
        v[i] = x;
    }
    return v;
}

static bool check(const char *name, const Mat &m, std::mt19937 &rng) {
    const int n = m.n, d = m.d, ng = n / 32;
    std::uniform_int_distribution<int> qd(-128, 127);
    std::uniform_real_distribution<float> xsd(1e-3f, 1e-1f);
    std::vector<int8_t> xq(n);
    std::vector<float> xs(ng);
    for (auto &v : xq) v = qd(rng);
    for (auto &v : xs) v = xsd(rng);

    std::vector<ap_uint<64> > wp = words(m.w.data(), m.w.size(), DEPTH_W), sp = words(m.s.data(), 2 * m.s.size(), DEPTH_WS),
                              xp = words(xq.data(), n, DEPTH_XQ);
    std::vector<float> xsp(xs), out(std::max<size_t>(d, DEPTH_OUT), NAN);
    xsp.resize(std::max<size_t>(ng, DEPTH_XS));
    matvec4(wp.data(), sp.data(), xp.data(), xsp.data(), out.data(), n, d, m.bits);

    double worst = 0, max_abs = 0;
    int bad = 0;
    for (int i = 0; i < d; i++) {
        double ref = 0, mag = 0;
        for (int g = 0; g < ng; g++) {
            long isum = 0;
            for (int k = 0; k < 32; k++) isum += (long)weight(m, (size_t)i * n + g * 32 + k) * xq[g * 32 + k];
            const double t = isum * half_ref(m.s[(size_t)i * ng + g]) * xs[g];
            ref += t;
            mag += std::fabs(t);
        }
        const double e = std::fabs((double)out[i] - ref), r = mag > 0 ? e / mag : e;
        if (!(r <= TOL)) bad++; // also catches NaN (row never written)
        worst = std::max(worst, r);
        max_abs = std::max(max_abs, e);
    }
    printf("%-22s n=%5d d=%5d w%d  max |err|/sum|terms| = %.2e  max |err| = %.2e  bad rows %d/%d  %s\n", name, n, d,
           m.bits, worst, max_abs, bad, d, bad ? "FAIL" : "ok");
    return bad == 0;
}

// Random weights at the given width; fp16 scales ~1e-3..6e-2 with random sign (Q4_0 scales are often negative),
// or with special=true, row i's scales all have exponent field i % 31: zero/subnormals (0) up to 65504 (30).
static Mat random_mat(int n, int d, int bits, std::mt19937 &rng, bool special = false) {
    Mat m{n, d, bits, std::vector<uint8_t>((size_t)d * n * bits / 8), std::vector<uint16_t>((size_t)d * n / 32)};
    std::uniform_int_distribution<int> byte(0, 255), mant(0, 1023), ex(5, 10), sign(0, 1);
    for (auto &b : m.w) b = byte(rng); // every nibble/byte value is a valid int4/int8
    for (size_t i = 0; i < m.s.size(); i++) {
        const int e = special ? (int)(i / (n / 32) % 31) : ex(rng);
        m.s[i] = (uint16_t)(sign(rng) << 15 | e << 10 | mant(rng));
    }
    return m;
}

// Walks the .qwb file the way tools/qwb_check.py does (layout: tools/gguf2qwb.py) and loads one matrix:
// kind 0..6 = wq, wk, wv, wo, w1 (gate), w2 (down), w3 (up) of `layer`; kind 7 = output head.
static bool load_qwb(FILE *f, int kind, int layer, Mat &m) {
    int32_t h[10]; // dim, hidden, L, heads, kv_heads, vocab, head_k, ctx, gs, emb_bits
    fseek(f, 8, SEEK_SET);
    if (fread(h, 4, 10, f) != 10) return false;
    const int dim = h[0], hidden = h[1], L = h[2], kv = dim / h[3] * h[4], head_k = h[6], emb_bits = h[9];
    const int index = kind * L + layer;
    long long off = 256;
    auto take = [&](long long bytes) { long long a = off; off += bytes + (-(off + bytes) & 63); return a; };
    for (int n : {dim, dim, dim, kv, kv})
        for (int l = 0; l < L; l++) take(4LL * n);
    take(4LL * dim);    // output_norm
    take(4LL * head_k); // head_ids
    const int shapes[7][2] = {{dim, dim}, {kv, dim}, {kv, dim}, {dim, dim}, {hidden, dim}, {dim, hidden}, {hidden, dim}};
    for (int i = 0; i <= 7 * L; i++) {
        const int rows = i < 7 * L ? shapes[i / L][0] : head_k, n = i < 7 * L ? shapes[i / L][1] : dim;
        const int bits = i < 7 * L ? 4 : emb_bits;
        const long long wo = take((long long)rows * n * bits / 8), so = take((long long)rows * n / 32 * 2);
        if (i != index) continue;
        m = Mat{n, rows, bits, std::vector<uint8_t>((size_t)rows * n * bits / 8), std::vector<uint16_t>((size_t)rows * n / 32)};
        fseek(f, (long)wo, SEEK_SET); // < 2 GiB
        if (fread(m.w.data(), 1, m.w.size(), f) != m.w.size()) return false;
        fseek(f, (long)so, SEEK_SET);
        return fread(m.s.data(), 2, m.s.size(), f) == m.s.size();
    }
    return false;
}

int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "all";
    std::mt19937 rng(1234);
    int fails = 0, count = 0;
    auto run = [&](const char *name, const Mat &m) { fails += !check(name, m, rng); count++; };

    if (mode == "lat") {
        run("random", random_mat(896, 4864, 4, rng));
        run("random", random_mat(896, 2432, 8, rng));
    } else {
        // odd d (write_out's lone last float), short rows (a single partial acc batch), max n, both widths
        static const int quick[][3] = {{96, 5, 4}, {96, 5, 8}, {896, 128, 4}, {896, 64, 8}, {4864, 32, 4}, {8192, 4, 4}};
        for (auto &s : quick) run("random", random_mat(s[0], s[1], s[2], rng));
        run("special fp16 scales", random_mat(64, 31, 4, rng, true));
        run("special fp16 scales", random_mat(64, 31, 8, rng, true));
    }
    if (mode == "all") {
        static const int shapes[][3] = {{896, 896, 4}, {896, 128, 4}, {896, 4864, 4}, {4864, 896, 4}, {896, 49152, 8}};
        for (auto &s : shapes) run("random", random_mat(s[0], s[1], s[2], rng));

        const char *env = getenv("QWB"); // run_hls.tcl points it at models/qwen/qwen2.5-coder-0.5b-q4-head49k.qwb
        const std::string path = env ? env : "(QWB not set)";
        if (FILE *f = fopen(path.c_str(), "rb")) {
            static const struct { const char *name; int kind, layer; } real[] = {
                {"qwb blk.0 attn_q", 0, 0},     {"qwb blk.0 attn_k", 1, 0},    {"qwb blk.23 ffn_gate", 4, 23},
                {"qwb blk.0 ffn_down", 5, 0},   {"qwb blk.12 ffn_up", 6, 12},  {"qwb output head", 7, 0}};
            for (auto &r : real) {
                Mat m;
                if (!load_qwb(f, r.kind, r.layer, m)) { printf("%s: read error in %s\n", r.name, path.c_str()); fails++; continue; }
                run(r.name, m);
            }
            fclose(f);
        } else {
            printf("real-matrix tests SKIPPED: %s not found\n", path.c_str());
        }
    }
    if (fails) printf("FAILED (%d of %d matrices)\n", fails, count);
    else printf("PASSED (%d matrices)\n", count);
    return fails ? 1 : 0;
}
