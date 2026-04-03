## constraints.xdc — Timing & Pin Constraints for FPGA Implementation
## ============================================================================
## Adjust pin LOC and IOSTANDARD to match YOUR specific board.
## Examples below are for a typical Zynq / Artix-7 board with 100 MHz clock.
## ============================================================================

## ---------------------------------------------------------------------------
## Clock (100 MHz oscillator)
## ---------------------------------------------------------------------------
## For Zybo / Zedboard:
# set_property PACKAGE_PIN L16 [get_ports sys_clk]
## For Basys3 / Nexys A7:
# set_property PACKAGE_PIN W5 [get_ports sys_clk]
## For Arty A7:
# set_property PACKAGE_PIN E3 [get_ports sys_clk]

# Uncomment and adjust for your board:
# set_property PACKAGE_PIN <YOUR_CLK_PIN> [get_ports sys_clk]
# set_property IOSTANDARD LVCMOS33 [get_ports sys_clk]

create_clock -period 10.000 -name sys_clk -waveform {0.000 5.000} [get_ports sys_clk]

## ---------------------------------------------------------------------------
## False paths for VIO/ILA (debug cores are asynchronous to user logic)
## Vivado typically handles this automatically, but add if timing fails:
## ---------------------------------------------------------------------------
# set_false_path -from [get_cells u_vio/*] -to [get_cells u_accel/*]
# set_false_path -from [get_cells u_accel/*] -to [get_cells u_ila/*]

## ---------------------------------------------------------------------------
## BRAM Inference Guidance (optional — Vivado usually infers correctly)
## ---------------------------------------------------------------------------
## If Vivado uses distributed RAM instead of Block RAM, add:
# set_property RAM_STYLE block [get_cells u_accel/input_bram_reg]
# set_property RAM_STYLE block [get_cells u_accel/output_bram_s0_reg]
# set_property RAM_STYLE block [get_cells u_accel/output_bram_s1_reg]
# set_property RAM_STYLE block [get_cells u_accel/output_bram_s2_reg]
# set_property RAM_STYLE block [get_cells u_accel/output_bram_s3_reg]
