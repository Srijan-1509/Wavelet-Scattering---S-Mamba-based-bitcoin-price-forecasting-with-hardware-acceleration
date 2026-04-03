// adaptive_wavelet_accelerator.v — Top-Level Adaptive-Precision Wavelet Scattering
// ============================================================================
// Synthesizable Verilog for Vivado FPGA (Xilinx Zynq)
//
// NOVELTY: Adaptive-precision wavelet scattering accelerator that dynamically
// selects between 8-bit and 16-bit precision based on input signal energy.
//   - Calm market (low energy)    → 8-bit:  saves power, faster
//   - Volatile market (high energy) → 16-bit: preserves accuracy
//
// Architecture:
//   1. Input data written to BRAM via external port
//   2. Energy detector monitors signal energy over sliding window
//   3. Wavelet filterbank computes 4-scale scattering coefficients
//   4. Precision mode is set by energy detector before each processing run
//   5. Output: 4 banks of scattering coefficients + performance metrics
//
// Interface:
//   - Write input data via input port before asserting 'start'
//   - Assert 'start' to begin processing
//   - Wait for 'done'
//   - Read scattering coefficients from output ports
//   - cycles, precision switches, power metric available after done
//
// Fixed-point: input/output Q8.8 (16-bit)
// ============================================================================

module adaptive_wavelet_accelerator #(
    parameter MAX_N         = 2000,
    parameter N_SCALES      = 4,
    parameter KERNEL_SIZE   = 16,
    parameter ENERGY_WINDOW = 64,
    parameter ENERGY_THRESH = 32'h00100000
)(
    input  wire        clk,
    input  wire        rst_n,
    
    // Control
    input  wire        start,
    output wire        done,
    
    // Energy threshold (programmable)
    input  wire [31:0] energy_threshold,
    
    // Input data write port
    input  wire        input_we,
    input  wire [10:0] input_addr,
    input  wire [15:0] input_wdata,    // Q8.8
    
    // Output data read port
    input  wire [10:0] output_addr,
    input  wire [1:0]  output_scale_sel, // Which scale to read (0-3)
    output wire [15:0] output_rdata,     // Q8.8 scattering coeff
    
    // Status / Metrics (mark_debug for ILA probing in Vivado)
    (* mark_debug = "true" *) output wire        precision_mode,   // Current: 0=8bit, 1=16bit
    (* mark_debug = "true" *) output wire [31:0] current_energy,   // Signal energy (debug)
    (* mark_debug = "true" *) output wire [31:0] total_cycles,     // Processing cycles
    output wire [31:0] cycles_8bit,      // Cycles if 8-bit mode
    output wire [31:0] cycles_16bit,     // Cycles if 16-bit mode
    (* mark_debug = "true" *) output reg  [15:0] precision_switches // Number of mode changes
);

    // =========================================================================
    // Input BRAM (dual-port: external write, engine read)
    // =========================================================================
    reg  [15:0] input_bram [0:2047];
    reg  [15:0] engine_rd_data;
    wire [10:0] engine_rd_addr;
    wire        engine_rd_en;
    
    // External write port
    always @(posedge clk) begin
        if (input_we)
            input_bram[input_addr] <= input_wdata;
    end
    
    // Engine read port
    always @(posedge clk) begin
        if (engine_rd_en)
            engine_rd_data <= input_bram[engine_rd_addr];
    end
    
    // =========================================================================
    // Output BRAMs (one per scale)
    // =========================================================================
    reg [15:0] output_bram_s0 [0:2047];
    reg [15:0] output_bram_s1 [0:2047];
    reg [15:0] output_bram_s2 [0:2047];
    reg [15:0] output_bram_s3 [0:2047];
    
    // Engine write port
    wire        eng_wr_en;
    wire [10:0] eng_wr_addr;
    wire [15:0] eng_wr_data;
    wire [1:0]  eng_wr_scale;
    
    always @(posedge clk) begin
        if (eng_wr_en) begin
            case (eng_wr_scale)
                2'd0: output_bram_s0[eng_wr_addr] <= eng_wr_data;
                2'd1: output_bram_s1[eng_wr_addr] <= eng_wr_data;
                2'd2: output_bram_s2[eng_wr_addr] <= eng_wr_data;
                2'd3: output_bram_s3[eng_wr_addr] <= eng_wr_data;
            endcase
        end
    end
    
    // External read port (muxed by scale select)
    reg [15:0] output_rd_reg;
    always @(posedge clk) begin
        case (output_scale_sel)
            2'd0: output_rd_reg <= output_bram_s0[output_addr];
            2'd1: output_rd_reg <= output_bram_s1[output_addr];
            2'd2: output_rd_reg <= output_bram_s2[output_addr];
            2'd3: output_rd_reg <= output_bram_s3[output_addr];
        endcase
    end
    assign output_rdata = output_rd_reg;
    
    // =========================================================================
    // Energy Detector — feeds into data path during input loading
    // =========================================================================
    wire        energy_precision_mode;
    wire [31:0] energy_val;
    
    wire [31:0] thresh_val;
    assign thresh_val = (energy_threshold != 32'd0) ? energy_threshold : ENERGY_THRESH;
    
    energy_detector #(
        .WINDOW_SIZE(ENERGY_WINDOW),
        .THRESHOLD(ENERGY_THRESH)
    ) u_energy (
        .clk              (clk),
        .rst_n            (rst_n),
        .data_valid       (input_we),
        .data_in          ($signed(input_wdata)),
        .energy_threshold (thresh_val),
        .precision_mode   (energy_precision_mode),
        .current_energy   (energy_val)
    );
    
    assign precision_mode = energy_precision_mode;
    assign current_energy = energy_val;
    
    // =========================================================================
    // Precision Switch Counter
    // =========================================================================
    reg prev_precision;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            precision_switches <= 16'd0;
            prev_precision     <= 1'b0;
        end else begin
            prev_precision <= energy_precision_mode;
            if (energy_precision_mode != prev_precision)
                precision_switches <= precision_switches + 16'd1;
        end
    end
    
    // =========================================================================
    // Wavelet Filterbank
    // =========================================================================
    wavelet_filterbank #(
        .MAX_N(MAX_N),
        .N_SCALES(N_SCALES),
        .KERNEL_SIZE(KERNEL_SIZE)
    ) u_filterbank (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (start),
        .precision_mode (energy_precision_mode),
        .done           (done),
        
        .input_data     (engine_rd_data),
        .input_addr     (engine_rd_addr),
        .input_rd_en    (engine_rd_en),
        
        .output_wr_en   (eng_wr_en),
        .output_addr    (eng_wr_addr),
        .output_data    (eng_wr_data),
        .output_scale   (eng_wr_scale),
        
        .cycles_8bit    (cycles_8bit),
        .cycles_16bit   (cycles_16bit),
        .total_cycles   (total_cycles)
    );

endmodule
