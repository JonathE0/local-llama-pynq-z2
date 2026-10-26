// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0 - ap_done (Read/TOW)
//        bit 1 - ap_ready (Read/TOW)
//        others - reserved
// 0x10 : Data signal of w
//        bit 31~0 - w[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of ws
//        bit 31~0 - ws[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of xq
//        bit 31~0 - xq[31:0] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of xs
//        bit 31~0 - xs[31:0] (Read/Write)
// 0x2c : reserved
// 0x30 : Data signal of out_r
//        bit 31~0 - out_r[31:0] (Read/Write)
// 0x34 : reserved
// 0x38 : Data signal of n
//        bit 31~0 - n[31:0] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of d
//        bit 31~0 - d[31:0] (Read/Write)
// 0x44 : reserved
// 0x48 : Data signal of wbits
//        bit 31~0 - wbits[31:0] (Read/Write)
// 0x4c : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XMATVEC4_CONTROL_ADDR_AP_CTRL    0x00
#define XMATVEC4_CONTROL_ADDR_GIE        0x04
#define XMATVEC4_CONTROL_ADDR_IER        0x08
#define XMATVEC4_CONTROL_ADDR_ISR        0x0c
#define XMATVEC4_CONTROL_ADDR_W_DATA     0x10
#define XMATVEC4_CONTROL_BITS_W_DATA     32
#define XMATVEC4_CONTROL_ADDR_WS_DATA    0x18
#define XMATVEC4_CONTROL_BITS_WS_DATA    32
#define XMATVEC4_CONTROL_ADDR_XQ_DATA    0x20
#define XMATVEC4_CONTROL_BITS_XQ_DATA    32
#define XMATVEC4_CONTROL_ADDR_XS_DATA    0x28
#define XMATVEC4_CONTROL_BITS_XS_DATA    32
#define XMATVEC4_CONTROL_ADDR_OUT_R_DATA 0x30
#define XMATVEC4_CONTROL_BITS_OUT_R_DATA 32
#define XMATVEC4_CONTROL_ADDR_N_DATA     0x38
#define XMATVEC4_CONTROL_BITS_N_DATA     32
#define XMATVEC4_CONTROL_ADDR_D_DATA     0x40
#define XMATVEC4_CONTROL_BITS_D_DATA     32
#define XMATVEC4_CONTROL_ADDR_WBITS_DATA 0x48
#define XMATVEC4_CONTROL_BITS_WBITS_DATA 32

