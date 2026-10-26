// Qwen2 / Qwen2.5 byte-level BPE tokenizer in plain C99. Contract: tokenizer.h; data file: tools/qwen_tok_export.py.
//
// Pipeline (identical to Hugging Face `tokenizers` for Qwen2.5):
//   text --split at added tokens (if allowed)--> segments --NFC--> Split regex (hand-coded below) --> ByteLevel
//   --> BPE (merges by rank, leftmost first on ties) --> ids.
// The Split regex is
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// and is matched with the same backtracking outcomes as Oniguruma (see pt_next). Letter / number / white-space classes
// and the NFC data are looked up in a run-length table from the data file (they are those of the `tokenizers` crate).
//
// Not thread-safe: a Tokenizer keeps scratch buffers and a lookup cache. Memory: ~6 MB for the tables + scratch
// proportional to the longest word / the largest segment that needs NFC.
#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { C_OTHER = 0, C_LETTER = 1, C_NUMBER = 2, C_SPACE = 3 };

typedef struct { uint32_t id, len; const char *s; } Added;
typedef struct { uint64_t key; uint32_t l, r; } HItem;           // heap entry: key = rank << 32 | position
typedef struct { uint8_t *p; size_t n, cap; } Buf;
typedef struct { uint32_t *p; size_t n, cap; } Vec;

struct Tokenizer {
    uint8_t *file;                                              // whole data file; the pointers below point into it
    uint32_t n_vocab, n_added, n_ids;
    const uint8_t *tok_len, *tok_blob;
    uint32_t *tok_off;                                          // n_vocab + 1 offsets into tok_blob
    uint32_t n_merges;
    const uint32_t *merges;                                     // (left, right, merged) per rank
    uint64_t *htab;                                             // (left, right) -> rank, open addressing
    uint32_t hmask;
    Added *added;
    uint8_t added_first[256];
    uint32_t n_uni, n_dec, n_comp;
    const uint32_t *uni, *dec_cp, *dec_off, *dec_pool, *comp;
    int32_t byte_id[256];
    uint8_t ascii_cls[128];
    uint32_t c_lo, c_len, c_val;                                // one-entry cache for uni_get
    // scratch
    int32_t *sym, *prv, *nxt;
    HItem *heap;
    size_t scap, hn, hcap;
    Buf nbuf;
    Vec rin, dec, ccc;
};

// ---------------------------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------------------------
static int buf_put(Buf *b, const void *p, size_t n) {
    if (b->cap - b->n < n) {
        size_t c = b->cap ? b->cap : 256;
        while (c - b->n < n) c *= 2;
        uint8_t *q = (uint8_t *)realloc(b->p, c);
        if (!q) return -1;
        b->p = q; b->cap = c;
    }
    memcpy(b->p + b->n, p, n);
    b->n += n;
    return 0;
}

static int vec_push(Vec *v, uint32_t x) {
    if (v->n == v->cap) {
        size_t c = v->cap ? v->cap * 2 : 256;
        uint32_t *q = (uint32_t *)realloc(v->p, c * sizeof *q);
        if (!q) return -1;
        v->p = q; v->cap = c;
    }
    v->p[v->n++] = x;
    return 0;
}

// Decode one UTF-8 character; returns its length. Invalid bytes become the pseudo code point 0x110000 + byte (length 1),
// so they pass through untouched (class "other", never changed by NFC).
static inline int u8dec(const uint8_t *s, size_t n, uint32_t *cp) {
    uint32_t c = s[0], d;
    if (c < 0x80) { *cp = c; return 1; }
    if (c >= 0xC2 && c < 0xE0 && n >= 2 && (s[1] & 0xC0) == 0x80) {
        *cp = (c & 0x1F) << 6 | (s[1] & 0x3F);
        return 2;
    }
    if (c >= 0xE0 && c < 0xF0 && n >= 3 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        d = (c & 0x0F) << 12 | (s[1] & 0x3F) << 6 | (s[2] & 0x3F);
        if (d >= 0x800 && (d < 0xD800 || d >= 0xE000)) { *cp = d; return 3; }
    }
    if (c >= 0xF0 && c < 0xF5 && n >= 4 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        d = (c & 0x07) << 18 | (s[1] & 0x3F) << 12 | (s[2] & 0x3F) << 6 | (s[3] & 0x3F);
        if (d >= 0x10000 && d < 0x110000) { *cp = d; return 4; }
    }
    *cp = 0x110000 + c;
    return 1;
}

static int put_cp(Buf *b, uint32_t c) {
    uint8_t u[4];
    int n;
    if (c < 0x80) { u[0] = (uint8_t)c; n = 1; }
    else if (c < 0x800) { u[0] = 0xC0 | c >> 6; u[1] = 0x80 | (c & 0x3F); n = 2; }
    else if (c < 0x10000) { u[0] = 0xE0 | c >> 12; u[1] = 0x80 | (c >> 6 & 0x3F); u[2] = 0x80 | (c & 0x3F); n = 3; }
    else if (c < 0x110000) {
        u[0] = 0xF0 | c >> 18; u[1] = 0x80 | (c >> 12 & 0x3F); u[2] = 0x80 | (c >> 6 & 0x3F); u[3] = 0x80 | (c & 0x3F);
        n = 4;
    } else { u[0] = (uint8_t)(c - 0x110000); n = 1; }
    return buf_put(b, u, (size_t)n);
}

// ---------------------------------------------------------------------------------------------------------------
// Unicode table: entry = start | payload << 21; payload = class (2 bits) | NFC-dirty << 2 | combining class << 3
// ---------------------------------------------------------------------------------------------------------------
static inline uint32_t uni_get(Tokenizer *t, uint32_t cp) {
    if (cp >= 0x110000) return 0;
    if (cp - t->c_lo < t->c_len) return t->c_val;
    uint32_t lo = 0, hi = t->n_uni;                              // last entry with start <= cp
    while (hi - lo > 1) {
        uint32_t mid = (lo + hi) / 2;
        if ((t->uni[mid] & 0x1FFFFF) <= cp) lo = mid; else hi = mid;
    }
    uint32_t start = t->uni[lo] & 0x1FFFFF;
    uint32_t next = lo + 1 < t->n_uni ? (t->uni[lo + 1] & 0x1FFFFF) : 0x110000;
    t->c_lo = start; t->c_len = next - start; t->c_val = t->uni[lo] >> 21;
    return t->c_val;
}

static inline unsigned cls_of(Tokenizer *t, uint32_t cp) {
    return cp < 128 ? t->ascii_cls[cp] : (uni_get(t, cp) & 3);
}

// ---------------------------------------------------------------------------------------------------------------
// NFC (UAX #15) on a region of code points; Hangul algorithmically, the rest from the tables
// ---------------------------------------------------------------------------------------------------------------
static int dec_find(const Tokenizer *t, uint32_t c) {
    uint32_t lo = 0, hi = t->n_dec;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (t->dec_cp[mid] < c) lo = mid + 1; else hi = mid;
    }
    return lo < t->n_dec && t->dec_cp[lo] == c ? (int)lo : -1;
}

static uint32_t compose_pair(const Tokenizer *t, uint32_t a, uint32_t b) {
    if (a >= 0x1100 && a < 0x1113 && b >= 0x1161 && b < 0x1176) return 0xAC00 + ((a - 0x1100) * 21 + (b - 0x1161)) * 28;
    if (a >= 0xAC00 && a < 0xAC00 + 11172 && (a - 0xAC00) % 28 == 0 && b > 0x11A7 && b < 0x11C3) return a + (b - 0x11A7);
    uint32_t lo = 0, hi = t->n_comp;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        const uint32_t *e = t->comp + 3 * mid;
        if (e[0] < a || (e[0] == a && e[1] < b)) lo = mid + 1; else hi = mid;
    }
    if (lo < t->n_comp && t->comp[3 * lo] == a && t->comp[3 * lo + 1] == b) return t->comp[3 * lo + 2];
    return 0;
}

// Appends NFC(in[0..n)) to *out (as UTF-8).
static int nfc_region(Tokenizer *t, const uint32_t *in, size_t n, Buf *out) {
    Vec *D = &t->dec, *K = &t->ccc;
    D->n = K->n = 0;
    for (size_t i = 0; i < n; i++) {                              // 1. canonical decomposition
        uint32_t c = in[i];
        int e;
        if (c >= 0xAC00 && c < 0xAC00 + 11172) {
            uint32_t s = c - 0xAC00;
            if (vec_push(D, 0x1100 + s / 588) || vec_push(D, 0x1161 + s % 588 / 28)) return -1;
            if (s % 28 && vec_push(D, 0x11A7 + s % 28)) return -1;
        } else if (c >= 0xC0 && (e = dec_find(t, c)) >= 0) {
            for (uint32_t k = t->dec_off[e]; k < t->dec_off[e + 1]; k++)
                if (vec_push(D, t->dec_pool[k])) return -1;
        } else if (vec_push(D, c)) return -1;
    }
    for (size_t i = 0; i < D->n; i++)
        if (vec_push(K, uni_get(t, D->p[i]) >> 3 & 0xFF)) return -1;
    uint32_t *d = D->p, *k = K->p;
    for (size_t i = 1; i < D->n; i++) {                           // 2. canonical ordering (stable insertion sort)
        if (!k[i]) continue;
        uint32_t c = d[i], cc = k[i];
        size_t j = i;
        while (j > 0 && k[j - 1] > cc) { d[j] = d[j - 1]; k[j] = k[j - 1]; j--; }
        d[j] = c; k[j] = cc;
    }
    size_t w = 0;                                                 // 3. canonical composition
    long starter = -1;
    for (size_t i = 0; i < D->n; i++) {
        uint32_t c = d[i], cc = k[i];
        if (starter >= 0 && ((long)w - 1 == starter || k[w - 1] < cc)) {
            uint32_t m = compose_pair(t, d[starter], c);
            if (m) { d[starter] = m; continue; }
        }
        if (!cc) starter = (long)w;
        d[w] = c; k[w] = cc; w++;
    }
    for (size_t i = 0; i < w; i++)
        if (put_cp(out, d[i])) return -1;
    return 0;
}

// Returns the NFC form of s[0..n): s itself when nothing changes (ASCII, or only NFC-stable characters), else t->nbuf.
// Normalization only works on a maximal run of non-ASCII characters plus the ASCII character before it (ASCII
// characters are starters that never combine backwards), so one far-away accent does not cost a whole-text pass.
static const uint8_t *normalize(Tokenizer *t, const uint8_t *s, size_t n, size_t *outn) {
    Buf *o = &t->nbuf;
    int started = 0;
    size_t k = 0;
    while (k < n) {
        size_t e = k;
        if (s[k] < 0x80) {
            while (e < n && s[e] < 0x80) e++;
            if (started && buf_put(o, s + k, e - k)) return NULL;
            k = e;
            continue;
        }
        int dirty = 0;
        while (e < n && s[e] >= 0x80) {
            uint32_t c;
            int l = u8dec(s + e, n - e, &c);
            if (uni_get(t, c) & 4) dirty = 1;
            e += (size_t)l;
        }
        if (!dirty) {
            if (started && buf_put(o, s + k, e - k)) return NULL;
            k = e;
            continue;
        }
        if (!started) { started = 1; o->n = 0; if (buf_put(o, s, k)) return NULL; }
        t->rin.n = 0;
        if (k > 0) { o->n--; if (vec_push(&t->rin, s[k - 1])) return NULL; }
        for (size_t i = k; i < e;) {
            uint32_t c;
            i += (size_t)u8dec(s + i, e - i, &c);
            if (vec_push(&t->rin, c)) return NULL;
        }
        if (nfc_region(t, t->rin.p, t->rin.n, o)) return NULL;
        k = e;
    }
    if (!started) { *outn = n; return s; }
    *outn = o->n;
    return o->p;
}

// ---------------------------------------------------------------------------------------------------------------
// Pre-tokenizer: returns the end of the piece that starts at s[i] (i < n)
// ---------------------------------------------------------------------------------------------------------------
// Bytes after an apostrophe that complete 's 't 'm 'd 're 've 'll (case-insensitive). With Oniguruma's (?i) the
// only non-ASCII spelling is U+017F (long s, folds to s); checked against `tokenizers` for every letter.
static size_t contraction(const uint8_t *s, size_t n) {
    if (n == 0) return 0;
    switch (s[0] | 0x20) {
    case 's': case 't': case 'm': case 'd': return 1;
    case 'r': case 'v': return n > 1 && (s[1] | 0x20) == 'e' ? 2 : 0;
    case 'l': return n > 1 && (s[1] | 0x20) == 'l' ? 2 : 0;
    }
    return n > 1 && s[0] == 0xC5 && s[1] == 0xBF ? 2 : 0;
}

static size_t scan_letters(Tokenizer *t, const uint8_t *s, size_t j, size_t n) {
    while (j < n) {
        uint32_t c;
        int l = u8dec(s + j, n - j, &c);
        if (cls_of(t, c) != C_LETTER) break;
        j += (size_t)l;
    }
    return j;
}

static size_t pt_next(Tokenizer *t, const uint8_t *s, size_t i, size_t n) {
    uint32_t c, d;
    int l = u8dec(s + i, n - i, &c), l2;
    unsigned k = cls_of(t, c);
    size_t j;
    if (c == '\'' && (j = contraction(s + i + 1, n - i - 1)) != 0) return i + 1 + j;           // 1: contractions
    if (k == C_LETTER) return scan_letters(t, s, i + (size_t)l, n);                          // 2: [^\r\n\p{L}\p{N}]?\p{L}+
    if (c != '\r' && c != '\n' && k != C_NUMBER && i + (size_t)l < n) {                      //    (one prefix character)
        j = i + (size_t)l;
        l2 = u8dec(s + j, n - j, &d);
        if (cls_of(t, d) == C_LETTER) return scan_letters(t, s, j + (size_t)l2, n);
    }
    if (k == C_NUMBER) return i + (size_t)l;                                                 // 3: \p{N}
    j = c == ' ' ? i + 1 : i;                                                                // 4: ' '?[^\s\p{L}\p{N}]+[\r\n]*
    if (j < n) {
        l2 = u8dec(s + j, n - j, &d);
        if (cls_of(t, d) == C_OTHER) {
            for (j += (size_t)l2; j < n; j += (size_t)l2) {
                l2 = u8dec(s + j, n - j, &d);
                if (cls_of(t, d) != C_OTHER) break;
            }
            while (j < n && (s[j] == '\r' || s[j] == '\n')) j++;
            return j;
        }
    }
    // c is white space. W = the maximal white-space run [i, q).
    size_t q = i, last = i, nl = 0;
    while (q < n) {
        l2 = u8dec(s + q, n - q, &d);
        if (cls_of(t, d) != C_SPACE) break;
        last = q;
        if (d == '\r' || d == '\n') nl = q + 1;
        q += (size_t)l2;
    }
    if (nl) return nl;                                  // 5: \s*[\r\n]+  -> up to the last newline of W
    if (q == n || last > i) return q == n ? q : last;   // 6: \s+(?!\S)   -> W, or W minus its last character
    return q;                                           // 7: \s+
}

// ---------------------------------------------------------------------------------------------------------------
// BPE
// ---------------------------------------------------------------------------------------------------------------
static inline uint32_t hash2(uint32_t a, uint32_t b) {
    uint32_t h = a * 0x9E3779B1u + b * 0x85EBCA6Bu;
    h ^= h >> 16; h *= 0x7FEB352Du;
    return h ^ h >> 15;
}

static inline int merge_rank(const Tokenizer *t, uint32_t a, uint32_t b) {
    uint64_t key = (uint64_t)a | (uint64_t)b << 18;
    for (uint32_t h = hash2(a, b) & t->hmask;; h = (h + 1) & t->hmask) {
        uint64_t e = t->htab[h];
        if (!e) return -1;
        if ((e & 0xFFFFFFFFFull) == key) return (int)(e >> 36 & 0x3FFFF);
    }
}

// Queue the pair (sym[p], sym[q]) if it has a merge. The heap is a min-heap on (rank, position): lowest rank first,
// leftmost on ties, like Hugging Face. Entries are never updated, only checked for staleness when popped.
static int pair_push(Tokenizer *t, int32_t p, int32_t q) {
    int r = merge_rank(t, (uint32_t)t->sym[p], (uint32_t)t->sym[q]);
    if (r < 0) return 0;
    if (t->hn == t->hcap) {
        size_t c = t->hcap ? t->hcap * 2 : 256;
        HItem *h = (HItem *)realloc(t->heap, c * sizeof *h);
        if (!h) return -1;
        t->heap = h; t->hcap = c;
    }
    HItem it = { (uint64_t)r << 32 | (uint32_t)p, (uint32_t)t->sym[p], (uint32_t)t->sym[q] }, *h = t->heap;
    size_t i = t->hn++;
    while (i > 0 && h[(i - 1) / 2].key > it.key) { h[i] = h[(i - 1) / 2]; i = (i - 1) / 2; }
    h[i] = it;
    return 0;
}

static HItem heap_pop(Tokenizer *t) {
    HItem *h = t->heap, top = h[0], it = h[--t->hn];
    size_t i = 0, n = t->hn;
    for (;;) {
        size_t c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && h[c + 1].key < h[c].key) c++;
        if (h[c].key >= it.key) break;
        h[i] = h[c]; i = c;
    }
    if (n) h[i] = it;
    return top;
}

// BPE of the bytes w[0..n); the result ids are left in t->sym[0..count). Returns count, or -1 on out of memory.
static int bpe_word(Tokenizer *t, const uint8_t *w, size_t n) {
    if (n > t->scap) {
        size_t c = n + n / 2 + 64;
        int32_t *a = (int32_t *)realloc(t->sym, 3 * c * sizeof *a);
        if (!a) return -1;
        t->sym = a; t->prv = a + c; t->nxt = a + 2 * c; t->scap = c;
    }
    int32_t *sym = t->sym, *prv = t->prv, *nxt = t->nxt;
    t->hn = 0;
    for (size_t i = 0; i < n; i++) { sym[i] = t->byte_id[w[i]]; prv[i] = (int32_t)i - 1; nxt[i] = (int32_t)i + 1; }
    for (size_t i = 0; i + 1 < n; i++)
        if (pair_push(t, (int32_t)i, (int32_t)i + 1)) return -1;
    while (t->hn) {
        HItem it = heap_pop(t);
        int32_t p = (int32_t)(uint32_t)it.key, q = nxt[p];
        if (sym[p] != (int32_t)it.l || q >= (int32_t)n || sym[q] != (int32_t)it.r) continue;   // stale entry
        sym[p] = (int32_t)t->merges[3 * (it.key >> 32) + 2];
        sym[q] = -1;
        nxt[p] = nxt[q];
        if (nxt[q] < (int32_t)n) prv[nxt[q]] = p;
        if (prv[p] >= 0 && pair_push(t, prv[p], p)) return -1;
        if (nxt[p] < (int32_t)n && pair_push(t, p, nxt[p])) return -1;
    }
    int cnt = 0;
    for (int32_t i = 0; i < (int32_t)n; i = nxt[i]) sym[cnt++] = sym[i];
    return cnt;
}

// ---------------------------------------------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------------------------------------------
static int enc_segment(Tokenizer *t, const uint8_t *s, size_t n, int *ids, int *cnt, int max_ids) {
    s = normalize(t, s, n, &n);
    if (!s) return -1;
    for (size_t i = 0; i < n;) {
        size_t j = pt_next(t, s, i, n);
        if (j == i + 1) {
            if (*cnt >= max_ids) return -1;
            ids[(*cnt)++] = t->byte_id[s[i]];
        } else {
            int m = bpe_word(t, s + i, j - i);
            if (m < 0 || m > max_ids - *cnt) return -1;
            memcpy(ids + *cnt, t->sym, (size_t)m * sizeof *ids);
            *cnt += m;
        }
        i = j;
    }
    return 0;
}

int tok_encode(Tokenizer *t, const char *text, int allow_special, int *ids, int max_ids) {
    const uint8_t *s = (const uint8_t *)text;
    size_t n = strlen(text), seg = 0;
    int cnt = 0;
    for (;;) {
        size_t p = n;
        const Added *hit = NULL;
        if (allow_special) {                                   // leftmost, then longest, added token
            for (size_t k = seg; k < n && !hit; k++) {
                if (!t->added_first[s[k]]) continue;
                for (uint32_t a = 0; a < t->n_added; a++) {
                    const Added *x = &t->added[a];
                    if (x->len <= n - k && memcmp(s + k, x->s, x->len) == 0 && (!hit || x->len > hit->len)) { hit = x; p = k; }
                }
            }
        }
        if (enc_segment(t, s + seg, p - seg, ids, &cnt, max_ids) < 0) return -1;
        if (!hit) return cnt;
        if (cnt >= max_ids) return -1;
        ids[cnt++] = (int)hit->id;
        seg = p + hit->len;
    }
}

const char *tok_decode(Tokenizer *t, int id, int *len) {
    if (id >= 0 && (uint32_t)id < t->n_vocab) {
        *len = t->tok_len[id];
        return (const char *)t->tok_blob + t->tok_off[id];
    }
    for (uint32_t a = 0; a < t->n_added; a++)
        if ((uint32_t)id == t->added[a].id) { *len = (int)t->added[a].len; return t->added[a].s; }
    *len = 0;
    return NULL;
}

int tok_id(Tokenizer *t, const char *added_token) {
    size_t n = strlen(added_token);
    for (uint32_t a = 0; a < t->n_added; a++)
        if (t->added[a].len == n && memcmp(t->added[a].s, added_token, n) == 0) return (int)t->added[a].id;
    return -1;
}

int tok_vocab_size(Tokenizer *t) { return (int)t->n_ids; }

// ---------------------------------------------------------------------------------------------------------------
// load / free
// ---------------------------------------------------------------------------------------------------------------
void tok_free(Tokenizer *t) {
    if (!t) return;
    free(t->file); free(t->tok_off); free(t->htab); free(t->added);
    free(t->sym); free(t->heap); free(t->nbuf.p); free(t->rin.p); free(t->dec.p); free(t->ccc.p);
    free(t);
}

Tokenizer *tok_load(const char *path) {
    Tokenizer *t = (Tokenizer *)calloc(1, sizeof *t);
    FILE *f = fopen(path, "rb");
    const char *why = "cannot open file";
    if (!t || !f) goto bad;
    long sz;
    if (fseek(f, 0, SEEK_END) || (sz = ftell(f)) < 64 || fseek(f, 0, SEEK_SET)) { why = "cannot read file"; goto bad; }
    t->file = (uint8_t *)malloc((size_t)sz);
    if (!t->file || fread(t->file, 1, (size_t)sz, f) != (size_t)sz) { why = "cannot read file"; goto bad; }
    fclose(f); f = NULL;
    why = "corrupt or wrong-version file (rebuild it with tools/qwen_tok_export.py)";
    const uint8_t *p = t->file, *end = p + sz;
    if (memcmp(p, "QWTOK001", 8)) goto bad;
    const uint32_t *h = (const uint32_t *)(p + 8);
    p += 8 + 48;
    for (int i = 0; i < 8; i++) if (h[i] > (uint32_t)sz) goto bad;
    t->n_vocab = h[0]; t->n_added = h[1]; t->n_merges = h[2]; t->n_uni = h[4]; t->n_dec = h[5]; t->n_comp = h[7];
    if (t->n_vocab > (1u << 18) || t->n_merges > (1u << 18) || t->n_uni < 1 || t->n_vocab < 256) goto bad;
#define TAKE(dst, type, count) do { size_t b_ = ((size_t)(count) * sizeof(type) + 3) & ~(size_t)3; \
        if (b_ > (size_t)(end - p)) { goto bad; } \
        dst = (const type *)p; p += b_; } while (0)
    TAKE(t->tok_len, uint8_t, t->n_vocab);
    TAKE(t->tok_blob, uint8_t, h[3]);
    TAKE(t->merges, uint32_t, 3 * t->n_merges);
    t->added = (Added *)calloc(t->n_added + 1, sizeof(Added));
    t->tok_off = (uint32_t *)malloc((t->n_vocab + 1) * sizeof(uint32_t));
    if (!t->added || !t->tok_off) goto bad;
    t->n_ids = t->n_vocab;
    for (uint32_t i = 0; i < t->n_added; i++) {
        const uint32_t *e;
        TAKE(e, uint32_t, 2);
        Added *a = &t->added[i];
        a->id = e[0]; a->len = e[1];
        TAKE(a->s, char, a->len);
        if (a->len == 0) goto bad;
        t->added_first[(uint8_t)a->s[0]] = 1;
        if (a->id + 1 > t->n_ids) t->n_ids = a->id + 1;
    }
    TAKE(t->uni, uint32_t, t->n_uni);
    TAKE(t->dec_cp, uint32_t, t->n_dec);
    TAKE(t->dec_off, uint32_t, t->n_dec + 1);
    TAKE(t->dec_pool, uint32_t, h[6]);
    TAKE(t->comp, uint32_t, 3 * t->n_comp);
#undef TAKE
    if (t->n_dec && t->dec_off[t->n_dec] > h[6]) goto bad;
    uint32_t off = 0;
    for (uint32_t i = 0; i < t->n_vocab; i++) { t->tok_off[i] = off; off += t->tok_len[i]; }
    t->tok_off[t->n_vocab] = off;
    if (off != h[3]) goto bad;
    for (int i = 0; i < 256; i++) t->byte_id[i] = -1;
    for (uint32_t i = 0; i < t->n_vocab; i++) if (t->tok_len[i] == 1) t->byte_id[t->tok_blob[t->tok_off[i]]] = (int32_t)i;
    for (int i = 0; i < 256; i++) if (t->byte_id[i] < 0) goto bad;
    uint32_t hs = 1024;
    while (hs < t->n_merges + t->n_merges / 2) hs *= 2;
    t->hmask = hs - 1;
    t->htab = (uint64_t *)calloc(hs, sizeof(uint64_t));
    if (!t->htab) goto bad;
    for (uint32_t r = 0; r < t->n_merges; r++) {
        uint32_t a = t->merges[3 * r], b = t->merges[3 * r + 1], m = t->merges[3 * r + 2];
        if (a >= t->n_vocab || b >= t->n_vocab || m >= t->n_vocab) goto bad;
        uint32_t x = hash2(a, b) & t->hmask;
        while (t->htab[x]) x = (x + 1) & t->hmask;
        t->htab[x] = (uint64_t)a | (uint64_t)b << 18 | (uint64_t)r << 36 | 1ull << 63;
    }
    for (uint32_t c = 0; c < 128; c++) t->ascii_cls[c] = (uint8_t)(uni_get(t, c) & 3);
    return t;
bad:
    fprintf(stderr, "tok_load(%s): %s\n", path, why);
    if (f) fclose(f);
    tok_free(t);
    return NULL;
}
