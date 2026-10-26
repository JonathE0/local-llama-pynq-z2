// C-sim / co-sim testbench for matvec(): compares against runq.c's matmul() on random data.
// The kernel keeps runq.c's float operation order, so the check is bit-exact (tolerance 0).
// Usage: no argument = all shapes (csim); "quick" = small shapes only (cosim).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "matvec.h"

// ---- verbatim from llama2.c/runq.c ----
int GS = 0; // group size global for quantization of the weights

typedef struct {
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
}
// ---- end verbatim ----

static std::vector<ap_uint<64> > pack(const std::vector<int8_t> &q) {
    std::vector<ap_uint<64> > p(q.size() / 8);
    for (size_t i = 0; i < p.size(); i++)
        for (int k = 0; k < 8; k++) p[i].range(8 * k + 7, 8 * k) = (uint8_t)q[8 * i + k];
    return p;
}

static bool test(int n, int d, int gs, std::mt19937 &rng) {
    std::uniform_int_distribution<int> qd(-128, 127);
    std::uniform_real_distribution<float> wsd(1e-4f, 2e-2f), xsd(1e-3f, 1e-1f);
    std::vector<int8_t> wq((size_t)d * n), xq(n);
    std::vector<float> ws((size_t)d * n / gs), xs(n / gs), ref(d), out(d, NAN);
    for (auto &v : wq) v = qd(rng);
    for (auto &v : xq) v = qd(rng);
    for (auto &v : ws) v = wsd(rng);
    for (auto &v : xs) v = xsd(rng);

    GS = gs;
    QuantizedTensor x = {xq.data(), xs.data()}, w = {wq.data(), ws.data()};
    matmul(ref.data(), &x, &w, n, d);

    // cosim copies `depth` elements of every m_axi buffer (pragmas in matvec.cpp), so pad up to that
    std::vector<ap_uint<64> > wp = pack(wq), xp = pack(xq);
    wp.resize(std::max<size_t>(wp.size(), 73728));
    ws.resize(std::max<size_t>(ws.size(), 9216));
    xp.resize(std::max<size_t>(xp.size(), 96));
    xs.resize(std::max<size_t>(xs.size(), 12));
    out.resize(std::max(d, 768), NAN);
    matvec(wp.data(), ws.data(), xp.data(), xs.data(), out.data(), n, d, gs);

    double max_abs = 0, max_rel = 0;
    int bad = 0;
    for (int i = 0; i < d; i++) {
        double e = std::fabs((double)out[i] - ref[i]);
        if (!(out[i] == ref[i])) bad++; // also catches NaN (row never written)
        if (e > max_abs) max_abs = e;
        if (ref[i] != 0 && e / std::fabs(ref[i]) > max_rel) max_rel = e / std::fabs(ref[i]);
    }
    printf("n=%5d d=%5d gs=%2d  max_abs_err=%.3g  max_rel_err=%.3g  mismatches=%d/%d  %s\n", n, d, gs, max_abs,
           max_rel, bad, d, bad ? "FAIL" : "ok");
    return bad == 0;
}

int main(int argc, char **argv) {
    static const int all[][3] = {{288, 288, 32}, {288, 768, 32}, {768, 288, 32}, {768, 768, 64},
                                 {768, 2048, 64}, {2048, 768, 64}, {768, 32000, 64},
                                 {96, 7, 32}}; // odd d: last row has no partner in acc()
    static const int quick[][3] = {{96, 7, 32}, {288, 288, 32}, {768, 768, 64}};
    const bool q = argc > 1 && !strcmp(argv[1], "quick");
    const int(*shapes)[3] = q ? quick : all;
    const int count = q ? 3 : 8;
    std::mt19937 rng(1234);
    int fails = 0;
    for (int i = 0; i < count; i++) fails += !test(shapes[i][0], shapes[i][1], shapes[i][2], rng);
    printf(fails ? "FAILED (%d shapes)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
