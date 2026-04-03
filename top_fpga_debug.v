// top_fpga_debug.v — FPGA Top-Level with VIO/ILA for Hardware Debug
// ============================================================================
// Wraps the adaptive wavelet accelerator with:
//   - VIO (Virtual I/O):  drive start, rst, threshold, input data from JTAG
//   - ILA (Integrated Logic Analyzer): capture internal waveforms
//   - BRAM: inferred from accelerator (Vivado auto-maps to Block RAM)
//
// Target: Xilinx Zynq / Artix-7 / Kintex-7 (Vivado 2020.2+)
//
// USAGE:
//   1. Add this as top module in Vivado
//   2. Generate VIO & ILA IP cores (see manual)
//   3. Synthesize → Implement → Generate Bitstream
//   4. Program FPGA → open Hardware Manager → interact via VIO/ILA
// ============================================================================

module top_fpga_debug #(
    parameter MAX_N         = 256,     // Use 256 for debug (saves BRAM)
    parameter N_SCALES      = 4,
    parameter KERNEL_SIZE   = 16,
    parameter ENERGY_WINDOW = 64,
    parameter ENERGY_THRESH = 32'h00008000
)(
    input  wire sys_clk       // Board oscillator (e.g., 100 MHz)
);

    // =========================================================================
    // Clock & Reset
    // =========================================================================
    wire clk;
    assign clk = sys_clk;    // Direct connect (or use MMCM/PLL if needed)

    // VIO-driven reset (active-low)
    wire vio_rst_n;

    // =========================================================================
    // VIO-driven control signals
    // =========================================================================
    wire        vio_start;
    wire [31:0] vio_energy_threshold;
    wire        vio_input_we;
    wire [10:0] vio_input_addr;
    wire [15:0] vio_input_wdata;
    wire [10:0] vio_output_addr;
    wire [1:0]  vio_output_scale_sel;

    // =========================================================================
    // Accelerator output signals (directly probed by ILA / readable by VIO)
    // =========================================================================
    wire        done;
    wire [15:0] output_rdata;
    wire        precision_mode;
    wire [31:0] current_energy;
    wire [31:0] total_cycles;
    wire [31:0] cycles_8bit;
    wire [31:0] cycles_16bit;
    wire [15:0] precision_switches;

    // =========================================================================
    // DUT: Adaptive Wavelet Scattering Accelerator
    // =========================================================================
    adaptive_wavelet_accelerator #(
        .MAX_N         (MAX_N),
        .N_SCALES      (N_SCALES),
        .KERNEL_SIZE   (KERNEL_SIZE),
        .ENERGY_WINDOW (ENERGY_WINDOW),
        .ENERGY_THRESH (ENERGY_THRESH)
    ) u_accel (
        .clk                (clk),
        .rst_n              (vio_rst_n),
        .start              (vio_start),
        .energy_threshold   (vio_energy_threshold),
        .input_we           (vio_input_we),
        .input_addr         (vio_input_addr),
        .input_wdata        (vio_input_wdata),
        .output_addr        (vio_output_addr),
        .output_scale_sel   (vio_output_scale_sel),
        .output_rdata       (output_rdata),
        .precision_mode     (precision_mode),
        .current_energy     (current_energy),
        .total_cycles       (total_cycles),
        .cycles_8bit        (cycles_8bit),
        .cycles_16bit       (cycles_16bit),
        .precision_switches (precision_switches)
    );

    // =========================================================================
    // VIO IP Core Instantiation
    // =========================================================================
    // Generate this IP in Vivado IP Catalog:
    //   IP Catalog → Debug & Verification → VIO (Virtual Input/Output)
    //   Component Name: vio_control
    //   Input Probe Count:  5  (FPGA→PC: read-only signals)
    //   Output Probe Count: 7  (PC→FPGA: writable controls)
    //
    //   OUTPUT probes (you drive FROM Vivado Hardware Manager):
    //     probe_out0 [0:0]  → rst_n           (reset, active-low)
    //     probe_out1 [0:0]  → start           (start pulse)
    //     probe_out2 [31:0] → energy_threshold
    //     probe_out3 [0:0]  → input_we        (write enable)
    //     probe_out4 [10:0] → input_addr
    //     probe_out5 [15:0] → input_wdata
    //     probe_out6 [10:0] → output_addr
    //
    //   INPUT probes (FPGA reports TO Vivado Hardware Manager):
    //     probe_in0 [0:0]   → done
    //     probe_in1 [0:0]   → precision_mode
    //     probe_in2 [31:0]  → current_energy
    //     probe_in3 [31:0]  → total_cycles
    //     probe_in4 [15:0]  → output_rdata
    //
    vio_control u_vio (
        .clk        (clk),

        // Input probes: FPGA → PC (read-only in Hardware Manager)
        .probe_in0  (done),                 // [0:0]
        .probe_in1  (precision_mode),       // [0:0]
        .probe_in2  (current_energy),       // [31:0]
        .probe_in3  (total_cycles),         // [31:0]
        .probe_in4  (output_rdata),         // [15:0]

        // Output probes: PC → FPGA (writable in Hardware Manager)
        .probe_out0 (vio_rst_n),            // [0:0]   reset
        .probe_out1 (vio_start),            // [0:0]   start
        .probe_out2 (vio_energy_threshold), // [31:0]  threshold
        .probe_out3 (vio_input_we),         // [0:0]   write enable
        .probe_out4 (vio_input_addr),       // [10:0]  input address
        .probe_out5 (vio_input_wdata),      // [15:0]  input data
        .probe_out6 (vio_output_addr)       // [10:0]  output address
    );

    // =========================================================================
    // ILA IP Core Instantiation
    // =========================================================================
    // Generate this IP in Vivado IP Catalog:
    //   IP Catalog → Debug & Verification → ILA (Integrated Logic Analyzer)
    //   Component Name: ila_monitor
    //   Number of Probes: 8
    //   Sample Data Depth: 4096 (or 8192 for longer capture)
    //   Trigger + Storage: Same probes
    //
    //   Probe widths:
    //     probe0  [0:0]   → start
    //     probe1  [0:0]   → done
    //     probe2  [0:0]   → precision_mode
    //     probe3  [31:0]  → current_energy
    //     probe4  [15:0]  → output_rdata
    //     probe5  [31:0]  → total_cycles
    //     probe6  [15:0]  → precision_switches
    //     probe7  [15:0]  → vio_input_wdata (to see data being written)
    //
    ila_monitor u_ila (
        .clk    (clk),
        .probe0 (vio_start),            // [0:0]
        .probe1 (done),                 // [0:0]
        .probe2 (precision_mode),       // [0:0]
        .probe3 (current_energy),       // [31:0]
        .probe4 (output_rdata),         // [15:0]
        .probe5 (total_cycles),         // [31:0]
        .probe6 (precision_switches),   // [15:0]
        .probe7 (vio_input_wdata)       // [15:0]
    );

endmodule
