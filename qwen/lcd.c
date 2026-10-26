// HD44780 driver for the 1602A on the qwen overlay's AXI GPIO (see lcd.h). Plain register writes through /dev/mem.
#include "lcd.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define LCD_GPIO_BASE 0x41200000

static volatile uint32_t *gpio; // [0] GPIO_DATA, [1] GPIO_TRI
static char line1[16];

static void wait_us(long us) { // busy-wait: the LCD needs ~40 us per byte, below the scheduler's sleep granularity
    struct timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    do { clock_gettime(CLOCK_MONOTONIC, &t); } while ((t.tv_sec - t0.tv_sec) * 1000000L + (t.tv_nsec - t0.tv_nsec) / 1000 < us);
}

static void nibble(int rs, int v) {
    uint32_t b = (rs ? 1u : 0u) | (uint32_t)(v & 0xF) << 2;
    gpio[0] = b;
    gpio[0] = b | 2; // E high for >= 450 ns
    wait_us(1);
    gpio[0] = b;     // falling edge latches the nibble
    wait_us(50);     // most instructions take 37 us
}

static void put(int rs, int v) { nibble(rs, v >> 4); nibble(rs, v); }

static void show(int row, const char *s) {
    put(0, row ? 0xC0 : 0x80); // DDRAM address of the row start
    for (int i = 0; i < 16; i++) { put(1, s[i]); }
}

int lcd_open(void) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("lcd: open /dev/mem (run as root)"); return -1; }
    void *p = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, LCD_GPIO_BASE);
    close(fd);
    if (p == MAP_FAILED) { perror("lcd: mmap"); return -1; }
    gpio = p;
    gpio[1] = 0; // all outputs
    gpio[0] = 0;
    wait_us(50000); // power-on settle
    nibble(0, 3); wait_us(5000); nibble(0, 3); wait_us(200); nibble(0, 3); // force 8-bit mode from any state
    nibble(0, 2);   // then switch to 4-bit
    put(0, 0x28);   // 4-bit, 2 lines, 5x8 font
    put(0, 0x0C);   // display on, no cursor
    put(0, 0x06);   // left to right, no shift
    put(0, 0x01);   // clear
    wait_us(2000);
    memset(line1, ' ', sizeof line1);
    return 0;
}

void lcd_text(const char *s, int len) {
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 0x80 && c < 0xC0) { continue; }        // UTF-8 continuation byte
        if (c >= 0xC0) { c = '?'; }                       // non-ASCII character
        else if (c < 0x20) { c = ' '; }                   // newline, tab
        else if (c == '\\') { c = '/'; }                  // HD44780 ROM: 0x5C is a yen sign
        else if (c >= 0x7E) { c = '-'; }                  // 0x7E is an arrow
        if (c == ' ' && line1[15] == ' ') { continue; }   // collapse whitespace runs
        memmove(line1, line1 + 1, 15);
        line1[15] = (char)c;
    }
    show(0, line1);
}

void lcd_status(const char *s) {
    char b[17];
    snprintf(b, sizeof b, "%-16s", s);
    show(1, b);
}
