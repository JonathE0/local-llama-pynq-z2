# 1602A LCD on the PYNQ-Z2 Raspberry Pi header (pins from PYNQ v3.0.1 boards/Pynq-Z2/base base.xdc), 4-bit mode.
# lcd[0] RS = header pin 11 (GPIO17), lcd[1] E = pin 13 (GPIO27), lcd[2..5] D4..D7 = pins 15, 16, 18, 22.
# LCD RW must be tied to GND so the 5 V LCD never drives these 3.3 V pins.
set_property -dict {PACKAGE_PIN U7  IOSTANDARD LVCMOS33} [get_ports {lcd[0]}]
set_property -dict {PACKAGE_PIN V7  IOSTANDARD LVCMOS33} [get_ports {lcd[1]}]
set_property -dict {PACKAGE_PIN U8  IOSTANDARD LVCMOS33} [get_ports {lcd[2]}]
set_property -dict {PACKAGE_PIN W6  IOSTANDARD LVCMOS33} [get_ports {lcd[3]}]
set_property -dict {PACKAGE_PIN Y7  IOSTANDARD LVCMOS33} [get_ports {lcd[4]}]
set_property -dict {PACKAGE_PIN F20 IOSTANDARD LVCMOS33} [get_ports {lcd[5]}]
