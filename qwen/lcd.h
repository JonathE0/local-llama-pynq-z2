// 1602A (HD44780) 16x2 character LCD on the qwen overlay's 6-bit AXI GPIO at 0x41200000.
// GPIO bit 0 = RS, bit 1 = E, bits 2..5 = D4..D7 (4-bit mode, RW tied to GND). Needs root and the qwen overlay.
#pragma once

int lcd_open(void);                     // 0 on success
void lcd_text(const char *s, int len);  // append text to line 1, which scrolls left; UTF-8 shown as '?'
void lcd_status(const char *s);         // replace line 2 (first 16 chars)
