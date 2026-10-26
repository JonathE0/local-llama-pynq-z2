/* Inference for Qwen2.5 (Qwen2 architecture) in pure C, in the style of llama2.c's runq.c:
   4-bit weights (int4 + fp16 scale per group of 32), activations quantized to int8 per group of 32 before each matmul.
   Weights: the .qwb file of tools/gguf2qwb.py. Numerical reference with the same arithmetic: tools/qwen_ref.py.
   -DFPGA: every matmul runs on the hls4/ matvec4 kernel (driver/); weights and quantized activations live in
   FPGA buffers. */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif
#include "tokenizer.h"
#include "lcd.h"
#ifdef FPGA
    #include "matvec_fpga.h"
#endif

#define GS 32            // group size of weight scales and activation scales
#define HEAD_BLK 2048    // output head rows per matmul call (keeps each FPGA buffer piece under 2 MiB)
#define IM_END 151645    // <|im_end|>
#define ENDOFTEXT 151643 // <|endoftext|>

// ----------------------------------------------------------------------------
// Transformer model

typedef struct { // the .qwb header after "QWB1" and the version
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size;
    int head_k;      // rows of the output head (logits only for these tokens)
    int max_seq_len, group_size, emb_bits;
    float rope_theta, rms_eps;
} Config;

typedef struct {
    const uint8_t *w;  // d*n*bits/8 bytes, row-major; 4-bit: element 2i of a row in the low nibble of byte i
    const uint16_t *s; // d*n/GS fp16 scales, one per group of 32 along a row
    int n, d, bits;    // in, out, weight bits (4 or 8)
#ifdef FPGA
    uint64_t pw, ps;   // physical addresses of w and s (copies in FPGA buffers)
#endif
} Matrix;

typedef struct {
    float *attn_norm, *ffn_norm; // (layer, dim)
    float *bq, *bk, *bv;         // (layer, dim), (layer, kv_dim), (layer, kv_dim)
    float *out_norm;             // (dim,)
    int32_t *head_ids;           // (head_k,) token id of each output-head row
    Matrix *wq, *wk, *wv, *wo;   // (layer,)
    Matrix *w1, *w2, *w3;        // (layer,) gate, down, up
    Matrix *head;                // output head (8-bit), in blocks of HEAD_BLK rows
    int n_head;                  // number of blocks
    Matrix emb;                  // full embedding table (8-bit); stays in the file mapping, one row read per token
} Weights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q; // activations
    int8_t *xq; float *xs;              // xb or hb quantized for the next matmul (FPGA buffer with -DFPGA)
    float *att;                         // (n_heads, ctx) attention scores
    float *logits;                      // (head_k,)
    float *key_cache, *value_cache;     // (layer, ctx, kv_dim)
    float *rope_cos, *rope_sin;         // (head_size/2,) for the current position
} RunState;

typedef struct {
    Config c;
    Weights w;
    RunState s;
    int ctx;          // KV cache length
    int fd;
    char *data;       // file mapping
    ssize_t file_size;
} Transformer;

static float f16(uint16_t h) { // fp16 bits -> float, exact (subnormals too)
    uint32_t e = (h >> 10) & 31, m = h & 1023, sign = (uint32_t)(h >> 15) << 31;
    if (e == 0) {
        if (m == 0) { e = 0; }
        else { int sh = __builtin_clz(m) - 21; m = (m << sh) & 1023; e = 113 - sh; } // normalize m * 2^-24
    } else {
        e += 112; // rebias 15 -> 127 (no inf/nan in scales)
    }
    union { uint32_t u; float f; } r = { .u = sign | (e << 23) | (m << 13) };
    return r.f;
}

#ifdef FPGA
// ----------------------------------------------------------------------------
// FPGA buffers (physically contiguous, non-cacheable; see driver/matvec_fpga.h)

static uint64_t xq_phys, xs_phys;  // physical addresses of RunState xq, xs
static float *fpga_out;            // matvec result, up to hidden_dim floats
static uint64_t fpga_out_phys;
static int fpga_check;             // FPGA_CHECK=1: also compute every matmul on the CPU and compare
static float *fpga_ref;
static double chk_abs, chk_rel;
static long chk_calls;

static void fpga_report(void) {
    fprintf(stderr, "FPGA_CHECK: %ld matmuls, max abs diff %g, max rel diff %g (outputs with |ref| > 1e-3)\n",
            chk_calls, chk_abs, chk_rel);
}

// Packed into chunks like runq/runq.c (zocl aligns each buffer up to 2 MiB, so one buffer per matrix would waste
// CMA), but first-fit decreasing: the biggest matrices go first, the small ones fill the gaps they leave (~0.1% lost
// instead of ~11% for in-order packing). A chunk allocation that fails (unmovable pages scattered in CMA) halves
// the chunk size, down to 2 MiB.
#define MAX_CHUNKS 512
static size_t fpga_chunk = (size_t)8 << 20;
static size_t fpga_left; // bytes still to be copied
static struct { char *p; uint64_t phys; size_t size, used; } chunk[MAX_CHUNKS];
static int n_chunks;

static char *fpga_carve(size_t bytes, uint64_t *phys) { // bytes: a multiple of 64
    int c = 0;
    while (c < n_chunks && chunk[c].size - chunk[c].used < bytes) { c++; }
    if (c == n_chunks) { // no room left anywhere: a new chunk
        if (n_chunks == MAX_CHUNKS) { fprintf(stderr, "fpga: too many chunks\n"); exit(EXIT_FAILURE); }
        size_t size = fpga_left < fpga_chunk ? fpga_left : fpga_chunk;
        if (size < bytes) { size = bytes; }
        while (!(chunk[c].p = fpga_alloc_try(size, &chunk[c].phys))) {
            if (fpga_chunk <= ((size_t)2 << 20) || fpga_chunk / 2 < bytes) {
                fpga_alloc(size, &chunk[c].phys); // fails again, and prints why
                exit(EXIT_FAILURE);
            }
            fpga_chunk /= 2;
            if (size > fpga_chunk) { size = fpga_chunk; }
        }
        chunk[c].size = size;
        chunk[c].used = 0;
        n_chunks++;
    }
    char *b = chunk[c].p + chunk[c].used;
    *phys = chunk[c].phys + chunk[c].used;
    chunk[c].used += bytes;
    fpga_left = fpga_left > bytes ? fpga_left - bytes : 0;
    return b;
}

static size_t pad64(size_t b) { return (b + 63) & ~(size_t)63; }
static size_t wbytes(const Matrix *m) { return (size_t)m->d * m->n * m->bits / 8; }
static size_t sbytes(const Matrix *m) { return (size_t)m->d * m->n / GS * 2; }
static size_t fpga_bytes(const Matrix *m) { return pad64(wbytes(m)) + pad64(sbytes(m)); }
static int bigger_first(const void *a, const void *b) {
    size_t x = fpga_bytes(*(Matrix *const *)a), y = fpga_bytes(*(Matrix *const *)b);
    return x < y ? 1 : x > y ? -1 : 0;
}

// replace a matrix that points into the file mapping by a copy in an FPGA buffer
static void fpga_move(Matrix *m) {
    uint64_t phys;
    char *b = fpga_carve(fpga_bytes(m), &phys);
    memcpy(b, m->w, wbytes(m));
    memcpy(b + pad64(wbytes(m)), m->s, sbytes(m));
    m->w = (const uint8_t *)b;
    m->s = (const uint16_t *)(b + pad64(wbytes(m)));
    m->pw = phys;
    m->ps = phys + pad64(wbytes(m));
}

static void fpga_load(Transformer *t) {
    Weights *w = &t->w;
    int L = t->c.n_layers, n = 0;
    Matrix **all = malloc((7 * L + w->n_head) * sizeof(Matrix *));
    Matrix *layer[] = { w->wq, w->wk, w->wv, w->wo, w->w1, w->w2, w->w3 };
    for (int a = 0; a < 7; a++)
        for (int l = 0; l < L; l++) { all[n++] = &layer[a][l]; }
    for (int b = 0; b < w->n_head; b++) { all[n++] = &w->head[b]; }
    qsort(all, n, sizeof(Matrix *), bigger_first);
    for (int i = 0; i < n; i++) { fpga_left += fpga_bytes(all[i]); }
    fprintf(stderr, "copying %.1f MiB of weights to FPGA buffers\n", fpga_left / 1048576.0);
    for (int i = 0; i < n; i++) { fpga_move(all[i]); }
    size_t got = 0;
    for (int c = 0; c < n_chunks; c++) { got += chunk[c].size; }
    fprintf(stderr, "%d FPGA buffers, %.1f MiB\n", n_chunks, got / 1048576.0);
    free(all);
    // the copies are made: let the kernel drop the file's pages (the embedding rows fault back in on demand)
    madvise(t->data, t->file_size, MADV_DONTNEED);
    posix_fadvise(t->fd, 0, 0, POSIX_FADV_DONTNEED);
}
#endif

// ----------------------------------------------------------------------------
// loading

static const void *take(const char *base, size_t *off, size_t bytes) { // every array starts on 64 bytes
    const void *p = base + *off;
    *off += (bytes + 63) & ~(size_t)63;
    return p;
}

static Matrix *matrices(const char *base, size_t *off, int count, int d, int n, int bits) {
    Matrix *m = malloc(count * sizeof(Matrix));
    for (int i = 0; i < count; i++) {
        m[i] = (Matrix){ .n = n, .d = d, .bits = bits };
        m[i].w = take(base, off, (size_t)d * n * bits / 8);
        m[i].s = take(base, off, (size_t)d * n / GS * 2);
    }
    return m;
}

static void read_checkpoint(Transformer *t, const char *path) {
    Config *p = &t->c;
    Weights *w = &t->w;
    FILE *file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", path); exit(EXIT_FAILURE); }
    char magic[4];
    uint32_t version;
    if (fread(magic, 4, 1, file) != 1 || fread(&version, 4, 1, file) != 1 || fread(p, sizeof(Config), 1, file) != 1
        || memcmp(magic, "QWB1", 4) || version != 1) {
        fprintf(stderr, "%s: not a .qwb version 1 file\n", path); exit(EXIT_FAILURE);
    }
    int hs = p->dim / p->n_heads, kv_dim = hs * p->n_kv_heads;
    if (p->group_size != GS || p->emb_bits != 8 || p->dim % p->n_heads || p->n_heads % p->n_kv_heads || hs % 2
        || p->dim % 128 || kv_dim % 16 || p->hidden_dim % 128) {
        fprintf(stderr, "%s: unsupported config (group %d, embedding %d-bit, dim %d, heads %d/%d)\n", path,
                p->group_size, p->emb_bits, p->dim, p->n_heads, p->n_kv_heads);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    t->file_size = ftell(file);
    fclose(file);
    t->fd = open(path, O_RDONLY);
    if (t->fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    t->data = mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
    if (t->data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

    // walk the file in the order of tools/gguf2qwb.py (per-layer vectors are multiples of 64 bytes, so contiguous)
    const char *b = t->data;
    size_t off = 256;
    int L = p->n_layers, dim = p->dim;
    w->attn_norm = (float *)take(b, &off, (size_t)L * dim * 4);
    w->ffn_norm = (float *)take(b, &off, (size_t)L * dim * 4);
    w->bq = (float *)take(b, &off, (size_t)L * dim * 4);
    w->bk = (float *)take(b, &off, (size_t)L * kv_dim * 4);
    w->bv = (float *)take(b, &off, (size_t)L * kv_dim * 4);
    w->out_norm = (float *)take(b, &off, (size_t)dim * 4);
    w->head_ids = (int32_t *)take(b, &off, (size_t)p->head_k * 4);
    w->wq = matrices(b, &off, L, dim, dim, 4);
    w->wk = matrices(b, &off, L, kv_dim, dim, 4);
    w->wv = matrices(b, &off, L, kv_dim, dim, 4);
    w->wo = matrices(b, &off, L, dim, dim, 4);
    w->w1 = matrices(b, &off, L, p->hidden_dim, dim, 4);
    w->w2 = matrices(b, &off, L, dim, p->hidden_dim, 4);
    w->w3 = matrices(b, &off, L, p->hidden_dim, dim, 4);
    Matrix head = matrices(b, &off, 1, p->head_k, dim, 8)[0];
    Matrix *emb = matrices(b, &off, 1, p->vocab_size, dim, 8);
    w->emb = *emb;
    free(emb);
    if (off != (size_t)t->file_size) {
        fprintf(stderr, "%s: size %lld, but the header describes %zu bytes\n", path, (long long)t->file_size, off);
        exit(EXIT_FAILURE);
    }
    // the head as blocks of HEAD_BLK rows (rows and their scales are contiguous in the file)
    w->n_head = (p->head_k + HEAD_BLK - 1) / HEAD_BLK;
    w->head = malloc(w->n_head * sizeof(Matrix));
    for (int i = 0; i < w->n_head; i++) {
        int r0 = i * HEAD_BLK;
        w->head[i] = (Matrix){ .n = dim, .d = p->head_k - r0 < HEAD_BLK ? p->head_k - r0 : HEAD_BLK, .bits = 8,
                               .w = head.w + (size_t)r0 * dim, .s = head.s + (size_t)r0 * dim / GS };
    }
}

static void *calloc_or_die(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    return p;
}

static void malloc_run_state(Transformer *t) {
    Config *p = &t->c;
    RunState *s = &t->s;
    int hs = p->dim / p->n_heads, kv_dim = hs * p->n_kv_heads;
    s->x = calloc_or_die(p->dim, sizeof(float));
    s->xb = calloc_or_die(p->dim, sizeof(float));
    s->xb2 = calloc_or_die(p->dim, sizeof(float));
    s->hb = calloc_or_die(p->hidden_dim, sizeof(float));
    s->hb2 = calloc_or_die(p->hidden_dim, sizeof(float));
    s->q = calloc_or_die(p->dim, sizeof(float));
    s->att = calloc_or_die((size_t)p->n_heads * t->ctx, sizeof(float));
    s->logits = calloc_or_die(p->head_k, sizeof(float));
    s->key_cache = calloc_or_die((size_t)p->n_layers * t->ctx * kv_dim, sizeof(float));
    s->value_cache = calloc_or_die((size_t)p->n_layers * t->ctx * kv_dim, sizeof(float));
    s->rope_cos = calloc_or_die(hs / 2, sizeof(float));
    s->rope_sin = calloc_or_die(hs / 2, sizeof(float));
    int max_n = p->hidden_dim > p->dim ? p->hidden_dim : p->dim;
#ifdef FPGA
    // one buffer: xq (max_n int8), xs (max_n/GS floats), then the matvec output (max_n floats)
    size_t xs_off = (max_n + 63) & ~63, out_off = xs_off + ((max_n / GS * sizeof(float) + 63) & ~63);
    uint64_t phys;
    char *buf = fpga_alloc(out_off + max_n * sizeof(float), &phys);
    if (!buf) { exit(EXIT_FAILURE); }
    s->xq = (int8_t *)buf;
    s->xs = (float *)(buf + xs_off);
    fpga_out = (float *)(buf + out_off);
    xq_phys = phys;
    xs_phys = phys + xs_off;
    fpga_out_phys = phys + out_off;
    const char *chk = getenv("FPGA_CHECK");
    if (chk && chk[0] == '1') {
        fpga_check = 1;
        fpga_ref = calloc_or_die(max_n, sizeof(float));
        atexit(fpga_report);
    }
#else
    s->xq = calloc_or_die(max_n, sizeof(int8_t));
    s->xs = calloc_or_die(max_n / GS, sizeof(float));
#endif
}

static void build_transformer(Transformer *t, const char *path, int ctx) {
#ifdef FPGA
    if (fpga_open() != 0) { exit(EXIT_FAILURE); } // prints its own error
#endif
    read_checkpoint(t, path);
    t->ctx = ctx < t->c.max_seq_len ? ctx : t->c.max_seq_len;
    malloc_run_state(t);
#ifdef FPGA
    fpga_load(t);
#endif
}
// ponytail: no free_transformer; the process exits right after generation and the OS (and fpga_close via exit)
// reclaims everything.

// ----------------------------------------------------------------------------
// neural net blocks

static void rmsnorm(float *o, const float *x, const float *weight, int size, float eps) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
    ss /= size;
    ss += eps;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) { o[j] = weight[j] * (ss * x[j]); }
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) { if (x[i] > max_val) { max_val = x[i]; } }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) { x[i] /= sum; }
}

// int8 per group of GS: scale = max|x| / 127, q = round(x / scale) (halves away from zero)
static void quantize(int8_t *q, float *s, const float *x, int n) {
    for (int g = 0; g < n / GS; g++) {
        float wmax = 0.0f;
        for (int i = 0; i < GS; i++) { wmax = fmaxf(wmax, fabsf(x[g * GS + i])); }
        float scale = wmax * (1.0f / 127.0f);
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        s[g] = scale;
        for (int i = 0; i < GS; i++) { q[g * GS + i] = (int8_t)roundf(x[g * GS + i] * inv); }
    }
}

#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
#define NEON_MATMUL
// one group of 32: int sums of w * x as 4 float lanes. int16 lanes take 2 products (int4: 2*8*127, int8: 2*128*127)
static inline float32x4_t dot4(const int8_t *w, int8x16x2_t x) { // x: even, odd activations (vld2q)
    int8x16_t p = vld1q_s8(w);
    int8x16_t lo = vshrq_n_s8(vshlq_n_s8(p, 4), 4), hi = vshrq_n_s8(p, 4); // even, odd weights
    int16x8_t t0 = vmull_s8(vget_low_s8(lo), vget_low_s8(x.val[0]));
    int16x8_t t1 = vmull_s8(vget_low_s8(hi), vget_low_s8(x.val[1]));
    t0 = vmlal_s8(t0, vget_high_s8(lo), vget_high_s8(x.val[0]));
    t1 = vmlal_s8(t1, vget_high_s8(hi), vget_high_s8(x.val[1]));
    return vcvtq_f32_s32(vpadalq_s16(vpaddlq_s16(t0), t1));
}

static inline float32x4_t dot8(const int8_t *w, int8x16_t x0, int8x16_t x1) {
    int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16);
    int16x8_t t0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
    int16x8_t t1 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
    t0 = vmlal_s8(t0, vget_high_s8(w0), vget_high_s8(x0));
    t1 = vmlal_s8(t1, vget_high_s8(w1), vget_high_s8(x1));
    return vcvtq_f32_s32(vpadalq_s16(vpaddlq_s16(t0), t1));
}

static inline float hsum(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(s, s), 0);
}

// group g+k of rows w0, w1 into accumulators acc0, acc1, scaled by lane `lane` of c0, c1
#define STEP4(k, acc0, acc1, c0, c1, lane) { \
    int8x16x2_t x = vld2q_s8(xq + (g + k) * GS); \
    acc0 = vmlaq_lane_f32(acc0, dot4(w0 + (g + k) * GS / 2, x), c0, lane); \
    acc1 = vmlaq_lane_f32(acc1, dot4(w1 + (g + k) * GS / 2, x), c1, lane); }
#define STEP8(k, acc0, acc1, c0, c1, lane) { \
    int8x16_t x0 = vld1q_s8(xq + (g + k) * GS), x1 = vld1q_s8(xq + (g + k) * GS + 16); \
    acc0 = vmlaq_lane_f32(acc0, dot8(w0 + (g + k) * GS, x0, x1), c0, lane); \
    acc1 = vmlaq_lane_f32(acc1, dot8(w1 + (g + k) * GS, x0, x1), c1, lane); }
#elif defined(__ARM_NEON)
#warning "the NEON matmul needs -mfpu=neon-fp16 -mfp16-format=ieee; building the plain C one"
#endif

// W (d,n) @ x (n,) -> out (d,):  out[i] = sum over groups g of (int sum of w*xq in g) * wscale * xscale[g]
static void matmul_cpu(float *out, const Matrix *m, const int8_t *xq, const float *xs) {
    const int n = m->n, ng = n / GS;
    const size_t row = (size_t)n * m->bits / 8; // bytes per row
    int i;
#ifdef NEON_MATMUL
    // two rows at a time (two independent chains for the in-order NEON pipeline), four groups per step (needs
    // ng % 4 == 0, checked at load); fp16 scales converted four at a time
    #pragma omp parallel for private(i)
    for (i = 0; i < m->d; i += 2) {
        int i1 = i + 1 < m->d ? i + 1 : i; // odd d: the last row is computed twice
        const int8_t *w0 = (const int8_t *)m->w + i * row, *w1 = (const int8_t *)m->w + i1 * row;
        const uint16_t *s0 = m->s + (size_t)i * ng, *s1 = m->s + (size_t)i1 * ng;
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, b0 = a0, b1 = a0;
        for (int g = 0; g < ng; g += 4) {
            float32x4_t xs4 = vld1q_f32(xs + g);
            float32x4_t c0 = vmulq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(s0 + g))), xs4);
            float32x4_t c1 = vmulq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(s1 + g))), xs4);
            float32x2_t c0l = vget_low_f32(c0), c0h = vget_high_f32(c0), c1l = vget_low_f32(c1), c1h = vget_high_f32(c1);
            if (m->bits == 4) {
                STEP4(0, a0, a1, c0l, c1l, 0) STEP4(1, b0, b1, c0l, c1l, 1)
                STEP4(2, a0, a1, c0h, c1h, 0) STEP4(3, b0, b1, c0h, c1h, 1)
            } else {
                STEP8(0, a0, a1, c0l, c1l, 0) STEP8(1, b0, b1, c0l, c1l, 1)
                STEP8(2, a0, a1, c0h, c1h, 0) STEP8(3, b0, b1, c0h, c1h, 1)
            }
        }
        out[i] = hsum(vaddq_f32(a0, b0));
        out[i1] = hsum(vaddq_f32(a1, b1));
    }
#else
    #pragma omp parallel for private(i)
    for (i = 0; i < m->d; i++) {
        const int8_t *w = (const int8_t *)m->w + i * row;
        const uint16_t *s = m->s + (size_t)i * ng;
        float val = 0.0f;
        for (int g = 0; g < ng; g++) {
            int32_t ival = 0;
            if (m->bits == 4) {
                for (int k = 0; k < GS / 2; k++) {
                    int8_t b = w[g * GS / 2 + k];
                    ival += ((int8_t)(b << 4) >> 4) * xq[g * GS + 2 * k] + (b >> 4) * xq[g * GS + 2 * k + 1];
                }
            } else {
                for (int k = 0; k < GS; k++) { ival += w[g * GS + k] * xq[g * GS + k]; }
            }
            val += ((float)ival * f16(s[g])) * xs[g];
        }
        out[i] = val;
    }
#endif
}

#ifdef FPGA
static void matmul(float *out, const Matrix *m, const int8_t *xq, const float *xs) {
    // same as matmul_cpu, but the kernel reads w, xq, xs from their FPGA buffers
    if (fpga_matvec4(m->pw, m->ps, xq_phys, xs_phys, fpga_out_phys, m->n, m->d, m->bits) != 0) { exit(EXIT_FAILURE); }
    memcpy(out, fpga_out, m->d * sizeof(float));
    if (fpga_check) {
        matmul_cpu(fpga_ref, m, xq, xs);
        for (int i = 0; i < m->d; i++) {
            double diff = fabs((double)out[i] - fpga_ref[i]);
            if (diff > chk_abs) { chk_abs = diff; }
            if (fabs(fpga_ref[i]) > 1e-3 && diff / fabs(fpga_ref[i]) > chk_rel) { chk_rel = diff / fabs(fpga_ref[i]); }
        }
        chk_calls++;
    }
}
#else
#define matmul matmul_cpu
#endif

static void rope(float *vec, int n_heads, int hs, const float *c, const float *s) {
    // NeoX / Hugging Face "rotate half": element i of a head pairs with element i + hs/2
    for (int h = 0; h < n_heads; h++) {
        float *v = vec + h * hs;
        for (int i = 0; i < hs / 2; i++) {
            float a = v[i], b = v[i + hs / 2];
            v[i] = a * c[i] - b * s[i];
            v[i + hs / 2] = b * c[i] + a * s[i];
        }
    }
}

static float *forward(Transformer *t, int token, int pos) {
    Config *p = &t->c;
    Weights *w = &t->w;
    RunState *s = &t->s;
    float *x = s->x;
    int dim = p->dim, hidden_dim = p->hidden_dim, hs = dim / p->n_heads;
    int kv_dim = hs * p->n_kv_heads, kv_mul = p->n_heads / p->n_kv_heads;

    // embedding row of this token, dequantized (the table stays in the file mapping)
    const int8_t *er = (const int8_t *)w->emb.w + (size_t)token * dim;
    const uint16_t *es = w->emb.s + (size_t)token * dim / GS;
    for (int i = 0; i < dim; i++) { x[i] = er[i] * f16(es[i / GS]); }

    // RoPE angles of this position, in double like tools/qwen_ref.py
    for (int i = 0; i < hs / 2; i++) {
        double a = pos * pow(p->rope_theta, -2.0 * i / hs);
        s->rope_cos[i] = (float)cos(a);
        s->rope_sin[i] = (float)sin(a);
    }

    for (int l = 0; l < p->n_layers; l++) {
        size_t loff = (size_t)l * t->ctx * kv_dim; // kv cache layer offset
        float *k = s->key_cache + loff + (size_t)pos * kv_dim;   // k, v go straight into the cache
        float *v = s->value_cache + loff + (size_t)pos * kv_dim;

        // attention rmsnorm, then q, k, v with biases
        rmsnorm(s->xb, x, w->attn_norm + l * dim, dim, p->rms_eps);
        quantize(s->xq, s->xs, s->xb, dim);
        matmul(s->q, &w->wq[l], s->xq, s->xs);
        matmul(k, &w->wk[l], s->xq, s->xs);
        matmul(v, &w->wv[l], s->xq, s->xs);
        for (int i = 0; i < dim; i++) { s->q[i] += w->bq[l * dim + i]; }
        for (int i = 0; i < kv_dim; i++) { k[i] += w->bk[l * kv_dim + i]; v[i] += w->bv[l * kv_dim + i]; }
        rope(s->q, p->n_heads, hs, s->rope_cos, s->rope_sin);
        rope(k, p->n_kv_heads, hs, s->rope_cos, s->rope_sin);

        // multihead attention (GQA: kv_mul query heads share a kv head)
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            const float *q = s->q + h * hs;
            float *att = s->att + (size_t)h * t->ctx;
            for (int ti = 0; ti <= pos; ti++) {
                const float *kt = s->key_cache + loff + (size_t)ti * kv_dim + (h / kv_mul) * hs;
                float score = 0.0f;
                for (int i = 0; i < hs; i++) { score += q[i] * kt[i]; }
                att[ti] = score / sqrtf(hs);
            }
            softmax(att, pos + 1);
            float *xb = s->xb + h * hs;
            memset(xb, 0, hs * sizeof(float));
            for (int ti = 0; ti <= pos; ti++) {
                const float *vt = s->value_cache + loff + (size_t)ti * kv_dim + (h / kv_mul) * hs;
                float a = att[ti];
                for (int i = 0; i < hs; i++) { xb[i] += a * vt[i]; }
            }
        }

        // output projection and residual
        quantize(s->xq, s->xs, s->xb, dim);
        matmul(s->xb2, &w->wo[l], s->xq, s->xs);
        for (int i = 0; i < dim; i++) { x[i] += s->xb2[i]; }

        // ffn: down(silu(gate(x)) * up(x)) and residual
        rmsnorm(s->xb, x, w->ffn_norm + l * dim, dim, p->rms_eps);
        quantize(s->xq, s->xs, s->xb, dim);
        matmul(s->hb, &w->w1[l], s->xq, s->xs);
        matmul(s->hb2, &w->w3[l], s->xq, s->xs);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            s->hb[i] = val * s->hb2[i];
        }
        quantize(s->xq, s->xs, s->hb, hidden_dim);
        matmul(s->xb, &w->w2[l], s->xq, s->xs);
        for (int i = 0; i < dim; i++) { x[i] += s->xb[i]; }
    }

    // final rmsnorm and the output head: logits of the head_ids tokens only
    rmsnorm(x, x, w->out_norm, dim, p->rms_eps);
    quantize(s->xq, s->xs, x, dim);
    for (int b = 0; b < w->n_head; b++) { matmul(s->logits + b * HEAD_BLK, &w->head[b], s->xq, s->xs); }
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Sampler (from llama2.c), over the head_k logits; returns a head row index

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

static int sample_argmax(const float *probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
    }
    return max_i;
}

static int sample_mult(const float *probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) { return i; }
    }
    return n - 1;
}

static int compare(const void *a, const void *b) {
    const ProbIndex *a_ = a, *b_ = b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

static int sample_topp(const float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) { probindex[n0].index = i; probindex[n0].prob = probabilities[i]; n0++; }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) { last_idx = i; break; }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) { return probindex[i].index; }
    }
    return probindex[last_idx].index;
}

static unsigned int random_u32(unsigned long long *state) { // xorshift rng
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }

static int sample(Sampler *sampler, float *logits) {
    if (sampler->temperature == 0.0f) { return sample_argmax(logits, sampler->vocab_size); }
    for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
    softmax(logits, sampler->vocab_size);
    float coin = random_f32(&sampler->rng_state);
    if (sampler->topp <= 0 || sampler->topp >= 1) { return sample_mult(logits, sampler->vocab_size, coin); }
    return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
}

// ----------------------------------------------------------------------------
// generation loop

static long time_in_ms(void) {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// prompt ids -> stream up to `steps` new tokens (stops at <|im_end|> / <|endoftext|> or a full context).
// tok == NULL: print the generated ids, comma separated. dump: every position's head logits (float32) appended.
static int use_lcd; // --lcd 1: mirror the output on the 1602A (line 1) with progress on line 2

static void generate(Transformer *t, Tokenizer *tok, Sampler *sampler, const int *prompt, int n_prompt, int steps,
                     FILE *dump) {
    long start = time_in_ms(), prompt_done = 0;
    int token = prompt[0], n_gen = 0, n_fwd_gen = 0, pos;
    char status[32];
    for (pos = 0; pos < t->ctx; pos++) {
        float *logits = forward(t, token, pos);
        if (dump) { fwrite(logits, sizeof(float), t->c.head_k, dump); }
        if (pos < n_prompt - 1) {
            if (use_lcd) { snprintf(status, sizeof status, "prompt %d/%d", pos + 1, n_prompt); lcd_status(status); }
            token = prompt[pos + 1];
            continue;
        }
        if (pos == n_prompt - 1) { prompt_done = time_in_ms(); } else { n_fwd_gen++; }
        int next = t->w.head_ids[sample(sampler, logits)];
        if (next == IM_END || next == ENDOFTEXT) { break; }
        if (tok) {
            int len;
            const char *piece = tok_decode(tok, next, &len);
            fwrite(piece, 1, len, stdout);
            if (use_lcd) {
                lcd_text(piece, len);
                snprintf(status, sizeof status, "%.1f tok/s %4dt", n_fwd_gen * 1000.0 / (time_in_ms() - prompt_done + 1e-9),
                         n_gen + 1);
                lcd_status(status);
            }
        } else {
            printf(n_gen ? ",%d" : "%d", next);
        }
        fflush(stdout);
        if (++n_gen >= steps) { break; }
        token = next;
    }
    long end = time_in_ms();
    if (use_lcd) {
        snprintf(status, sizeof status, "%.2f tok/s done", n_fwd_gen ? n_fwd_gen * 1000.0 / (end - prompt_done + 1e-9) : 0.0);
        lcd_status(status);
    }
    printf("\n");
    if (pos == t->ctx) { fprintf(stderr, "(context of %d tokens full)\n", t->ctx); }
    fprintf(stderr, "prompt: %d tokens in %.2f s (%.2f tok/s); generated: %d tokens, %.2f tok/s\n", n_prompt,
            (prompt_done - start) / 1000.0, n_prompt * 1000.0 / (prompt_done - start + 1e-9), n_gen,
            n_fwd_gen ? n_fwd_gen * 1000.0 / (end - prompt_done + 1e-9) : 0.0);
}

// ----------------------------------------------------------------------------
// CLI

static void error_usage(void) {
    fprintf(stderr, "Usage:   qwen <model.qwb> [options]\n");
    fprintf(stderr, "Example: qwen qwen2.5-coder-0.5b-q4-head49k.qwb -i \"Write a function that reverses a string\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 0.7 (0 = greedy)\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.8\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    max new tokens, default 256\n");
    fprintf(stderr, "  -c <int>    context length (KV cache), default 1024\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -m <string> chat|raw, default chat (Qwen chat template around the prompt; raw = as is)\n");
    fprintf(stderr, "  -z <string> tokenizer file, default qwen_tokenizer.bin next to the model\n");
    fprintf(stderr, "  --ids <list>     prompt as comma-separated token ids (no tokenizer; prints generated ids)\n");
    fprintf(stderr, "  --logits <file>  write every position's head logits (float32 x head_k) to file\n");
    fprintf(stderr, "  --lcd 1          also show the output on the 1602A LCD (root, qwen overlay with the LCD GPIO)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL, *tokenizer_path = NULL, *prompt = "", *mode = "chat", *ids_arg = NULL;
    char *logits_path = NULL;
    float temperature = 0.7f, topp = 0.8f;
    int steps = 256, ctx = 1024;
    unsigned long long rng_seed = 0;

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-') { error_usage(); }
        if (!strcmp(argv[i], "--ids")) { ids_arg = argv[i + 1]; continue; }
        if (!strcmp(argv[i], "--logits")) { logits_path = argv[i + 1]; continue; }
        if (!strcmp(argv[i], "--lcd")) { use_lcd = atoi(argv[i + 1]); continue; }
        if (strlen(argv[i]) != 2) { error_usage(); }
        switch (argv[i][1]) {
            case 't': temperature = atof(argv[i + 1]); break;
            case 'p': topp = atof(argv[i + 1]); break;
            case 's': rng_seed = strtoull(argv[i + 1], NULL, 10); break;
            case 'n': steps = atoi(argv[i + 1]); break;
            case 'c': ctx = atoi(argv[i + 1]); break;
            case 'i': prompt = argv[i + 1]; break;
            case 'm': mode = argv[i + 1]; break;
            case 'z': tokenizer_path = argv[i + 1]; break;
            default: error_usage();
        }
    }
    if (rng_seed == 0) { rng_seed = (unsigned int)time(NULL); }
    if (temperature < 0.0f) { temperature = 0.0f; }
    if (topp < 0.0f || 1.0f < topp) { topp = 0.8f; }
    if (steps < 1 || ctx < 2) { error_usage(); }
    if (strcmp(mode, "chat") && strcmp(mode, "raw")) { error_usage(); }
    if (use_lcd && lcd_open() != 0) { exit(EXIT_FAILURE); } // prints its own error

    // prompt token ids: from --ids, or the tokenizer (chat template by default)
    Tokenizer *tok = NULL;
    int *ids, n_ids = 0;
    if (ids_arg) {
        ids = calloc_or_die(strlen(ids_arg) / 2 + 1, sizeof(int));
        for (char *c = ids_arg, *end; *c; c = *end ? end + 1 : end) {
            long v = strtol(c, &end, 10);
            if (end == c || (*end && *end != ',') || v < 0) { fprintf(stderr, "bad --ids\n"); exit(EXIT_FAILURE); }
            ids[n_ids++] = (int)v;
        }
    } else {
        char default_tok[4096];
        if (!tokenizer_path) { // qwen_tokenizer.bin in the model's directory
            const char *slash = strrchr(checkpoint_path, '/');
            int dir = slash ? (int)(slash - checkpoint_path + 1) : 0;
            snprintf(default_tok, sizeof default_tok, "%.*sqwen_tokenizer.bin", dir, checkpoint_path);
            tokenizer_path = default_tok;
        }
        tok = tok_load(tokenizer_path);
        if (!tok) { exit(EXIT_FAILURE); } // prints its own error
        const char *tmpl = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                           "<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
        size_t len = strlen(tmpl) + strlen(prompt) + 1;
        char *text = calloc_or_die(len, 1);
        snprintf(text, len, strcmp(mode, "chat") ? "%s" : tmpl, prompt);
        ids = calloc_or_die(len, sizeof(int)); // never more ids than bytes
        // special tokens in the text become their ids, like HF's apply_chat_template + encode
        n_ids = tok_encode(tok, text, 1, ids, (int)len);
        free(text);
        if (n_ids < 1) { fprintf(stderr, "empty prompt\n"); exit(EXIT_FAILURE); }
    }

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path, ctx);
    for (int i = 0; i < n_ids; i++) {
        if (ids[i] >= transformer.c.vocab_size) { fprintf(stderr, "token id %d out of range\n", ids[i]); exit(EXIT_FAILURE); }
    }
    if (n_ids >= transformer.ctx) { fprintf(stderr, "prompt (%d tokens) does not fit -c %d\n", n_ids, transformer.ctx); exit(EXIT_FAILURE); }

    Sampler sampler = { .vocab_size = transformer.c.head_k, .temperature = temperature, .topp = topp,
                        .rng_state = rng_seed, .probindex = calloc_or_die(transformer.c.head_k, sizeof(ProbIndex)) };
    FILE *dump = NULL;
    if (logits_path && !(dump = fopen(logits_path, "wb"))) { fprintf(stderr, "cannot write %s\n", logits_path); exit(EXIT_FAILURE); }
    generate(&transformer, tok, &sampler, ids, n_ids, steps, dump);
    if (dump) { fclose(dump); }
#ifdef FPGA
    fpga_close();
#endif
    return 0;
}
