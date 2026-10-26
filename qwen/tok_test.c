// Test CLI for tokenizer.c.
//   tok_test <qwen_tokenizer.bin> enc [-s] [-t] [file]   encode; NUL-separated cases (a file without NUL is one case);
//                                                        one line of space-separated ids per case. -s: allow special tokens
//   tok_test <qwen_tokenizer.bin> dec [file]             one line of ids per case in, decoded bytes + NUL per line out
//   -t (enc): timing, memory and token counts on stderr.   file "-" or none: stdin.
//   build: gcc -O2 -std=c99 -o tok_test tok_test.c tokenizer.c   (POSIX: add -D_POSIX_C_SOURCE=200809L, or use gnu99)
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

static char *slurp(const char *path, size_t *len) {
    FILE *f = (!path || !strcmp(path, "-")) ? stdin : fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    size_t cap = 1 << 16, n = 0, r;
    char *b = (char *)malloc(cap + 1);
    while ((r = fread(b + n, 1, cap - n, f)) > 0) {
        n += r;
        if (n == cap) b = (char *)realloc(b, (cap *= 2) + 1);
        if (!b) exit(1);
    }
    b[n] = 0;
    if (f != stdin) fclose(f);
    *len = n;
    return b;
}

static long rss_kb(const char *key) {                        // Linux only: VmRSS / VmHWM from /proc/self/status
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long v = -1;
    size_t kl = strlen(key);
    while (f && fgets(line, sizeof line, f))
        if (!strncmp(line, key, kl)) { v = atol(line + kl + 1); break; }
    if (f) fclose(f);
    return v;
}

int main(int argc, char **argv) {
    int special = 0, timing = 0, dec = 0;
    const char *bin = NULL, *file = NULL;
    for (int i = 1, pos = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-s")) special = 1;
        else if (!strcmp(argv[i], "-t")) timing = 1;
        else if (pos == 0) { bin = argv[i]; pos++; }
        else if (pos == 1) { dec = !strcmp(argv[i], "dec"); if (!dec && strcmp(argv[i], "enc")) bin = NULL; pos++; }
        else file = argv[i];
    }
    if (!bin) { fprintf(stderr, "usage: %s <qwen_tokenizer.bin> enc [-s] [-t] [file] | dec [file]\n", argv[0]); return 2; }
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    static char obuf[1 << 16];
    setvbuf(stdout, obuf, _IOFBF, sizeof obuf);

    long rss0 = rss_kb("VmRSS:");
    clock_t c0 = clock();
    Tokenizer *t = tok_load(bin);
    if (!t) return 1;
    double load_s = (double)(clock() - c0) / CLOCKS_PER_SEC;
    long rss1 = rss_kb("VmRSS:");

    size_t len, nbytes = 0;
    char *in = slurp(file, &len);
    int rc = 0;
    if (dec) {
        for (char *p = in; p < in + len;) {                  // one line per case
            char *e = (char *)memchr(p, '\n', (size_t)(in + len - p));
            if (!e) e = in + len;
            *e = 0;
            for (char *q = p; *q;) {
                char *end;
                long id = strtol(q, &end, 10);
                if (end == q) break;
                int n;
                const char *b = tok_decode(t, (int)id, &n);
                if (!b) { fprintf(stderr, "bad id %ld\n", id); rc = 1; }
                else fwrite(b, 1, (size_t)n, stdout);
                q = end;
            }
            fputc(0, stdout);
            p = e + 1;
        }
    } else {
        size_t cap = len + 16;
        int *ids = (int *)malloc(cap * sizeof *ids);
        long total = 0;
        double enc_s = 0;
        for (char *p = in; p < in + len; p += strlen(p) + 1) {
            int n;
            clock_t c1 = clock();
            while ((n = tok_encode(t, p, special, ids, (int)cap)) < 0) ids = (int *)realloc(ids, (cap *= 2) * sizeof *ids);
            enc_s += (double)(clock() - c1) / CLOCKS_PER_SEC;
            nbytes += strlen(p);
            total += n;
            for (int i = 0; i < n; i++) fprintf(stdout, i ? " %d" : "%d", ids[i]);
            fputc('\n', stdout);
        }
        if (timing)
            fprintf(stderr, "load %.3fs  encode %.3fs  %zu bytes -> %ld ids  (%.0f KB/s)\n", load_s, enc_s, nbytes, total,
                    enc_s > 0 ? nbytes / 1024.0 / enc_s : 0.0);
        free(ids);
    }
    if (timing && rss1 >= 0)
        fprintf(stderr, "rss: tokenizer %ld KB (file load), peak after run %ld KB\n", rss1 - rss0, rss_kb("VmHWM:"));
    free(in);
    tok_free(t);
    return rc;
}
