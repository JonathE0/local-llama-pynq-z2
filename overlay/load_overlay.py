# Load the matvec overlay and set the kernel clock. Run as root with the PYNQ environment (see README).
# Vivado timed the design at 150 MHz; the closest the PS PLL gets is 142.86 MHz (the overlay load alone leaves 125).
import os

from pynq import Clocks, Overlay

Overlay(os.path.join(os.path.dirname(os.path.abspath(__file__)), "matvec.bit"))
Clocks.fclk0_mhz = 142.857
# Page cache left in the 128 MiB CMA pool fragments it; stories110M needs ~116 MB of it right after this.
# ponytail: if a later run still fails in fpga_alloc, drop caches again (or grow CMA with a cma= boot arg).
os.sync()
with open("/proc/sys/vm/drop_caches", "w") as f:
    f.write("3")
print("matvec overlay loaded, fclk0 =", Clocks.fclk0_mhz, "MHz")
