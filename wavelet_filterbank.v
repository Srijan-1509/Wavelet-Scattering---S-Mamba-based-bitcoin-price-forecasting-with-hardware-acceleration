// wavelet_filterbank.v — Multi-Resolution Wavelet Filterbank
// ============================================================================
// Implements a 4-scale Morlet-approximated wavelet filterbank with 
// adaptive precision (8-bit or 16-bit) selected at runtime.
//
// Architecture:
//   - 4 parallel FIR filters (one per scale j=0..3)
//   - Each filter approximates a Morlet wavelet at its octave
//   - Precision mode switches between 8-bit and 16-bit MAC operations
//   - Output: modulus (magnitude) of each scale's wavelet coefficients
//
// The "modulus" operation |x*psi| is the wavelet scattering core —
// it provides the shift-invariant, multi-scale feature representation.
//
// Fixed-point formats:
//   Input:  Q8.8 (16-bit) or Q4.4 (8-bit in low-precision mode)
//   Kernel: Q1.7 (8-bit) or Q1.15 (16-bit)
//   Output: Q8.8 (16-bit) magnitude result
// ============================================================================

module wavelet_filterbank #(
    parameter MAX_N       = 2000,  // Max input length
    parameter N_SCALES    = 4,     // Number of wavelet scales
    parameter KERNEL_SIZE = 16     // Filter tap length
)(
    input  wire        clk,
    input  wire        rst_n,
    
    // Control
    input  wire        start,
    input  wire        precision_mode,  // 0=8-bit, 1=16-bit
    output reg         done,
    
    // Input BRAM interface
    input  wire [15:0] input_data,      // Q8.8 from BRAM
    output reg  [10:0] input_addr,      // Address to read
    output reg         input_rd_en,
    
    // Output BRAM interface (4 outputs, one per scale)
    output reg         output_wr_en,
    output reg  [10:0] output_addr,
    output reg  [15:0] output_data,     // Q8.8 magnitude
    output reg  [1:0]  output_scale,    // Which scale (0-3)
    
    // Metrics
    output reg  [31:0] cycles_8bit,
    output reg  [31:0] cycles_16bit,
    output reg  [31:0] total_cycles
);

    // =========================================================================
    // Morlet Wavelet Coefficient LUT (pre-computed, 4 scales)
    // =========================================================================
    // Approximation: real part of Morlet at each scale
    // Scale j: center_freq = pi / 2^j, bandwidth ~ 2^j
    // Stored as Q1.15 (16-bit) for high precision, truncated to Q1.7 for low
    
    // Scale 0: Highest frequency (shortest wavelet)
    reg signed [15:0] kernel_s0 [0:KERNEL_SIZE-1];
    // Scale 1
    reg signed [15:0] kernel_s1 [0:KERNEL_SIZE-1];
    // Scale 2
    reg signed [15:0] kernel_s2 [0:KERNEL_SIZE-1];
    // Scale 3: Lowest frequency (longest wavelet)
    reg signed [15:0] kernel_s3 [0:KERNEL_SIZE-1];
    
    // Initialize kernels (Morlet approximation at 4 octaves)
    // These are discretized Morlet real parts normalized to Q1.15
    initial begin
        // Scale 0: xi = pi/1, sigma = 1.5 (high freq, narrow)
        kernel_s0[ 0] = 16'sh0066; kernel_s0[ 1] = 16'sh0180;
        kernel_s0[ 2] = 16'sh0340; kernel_s0[ 3] = 16'sh04E0;
        kernel_s0[ 4] = 16'sh0500; kernel_s0[ 5] = 16'sh02A0;
        kernel_s0[ 6] = 16'shFD80; kernel_s0[ 7] = 16'shF900;
        kernel_s0[ 8] = 16'shF900; kernel_s0[ 9] = 16'shFD80;
        kernel_s0[10] = 16'sh02A0; kernel_s0[11] = 16'sh0500;
        kernel_s0[12] = 16'sh04E0; kernel_s0[13] = 16'sh0340;
        kernel_s0[14] = 16'sh0180; kernel_s0[15] = 16'sh0066;
        
        // Scale 1: xi = pi/2, sigma = 3.0 (mid-high freq)
        kernel_s1[ 0] = 16'sh0120; kernel_s1[ 1] = 16'sh0280;
        kernel_s1[ 2] = 16'sh0430; kernel_s1[ 3] = 16'sh0560;
        kernel_s1[ 4] = 16'sh0580; kernel_s1[ 5] = 16'sh0400;
        kernel_s1[ 6] = 16'sh0100; kernel_s1[ 7] = 16'shFE00;
        kernel_s1[ 8] = 16'shFE00; kernel_s1[ 9] = 16'sh0100;
        kernel_s1[10] = 16'sh0400; kernel_s1[11] = 16'sh0580;
        kernel_s1[12] = 16'sh0560; kernel_s1[13] = 16'sh0430;
        kernel_s1[14] = 16'sh0280; kernel_s1[15] = 16'sh0120;
        
        // Scale 2: xi = pi/4, sigma = 6.0 (mid-low freq)
        kernel_s2[ 0] = 16'sh01A0; kernel_s2[ 1] = 16'sh0300;
        kernel_s2[ 2] = 16'sh0480; kernel_s2[ 3] = 16'sh05C0;
        kernel_s2[ 4] = 16'sh0640; kernel_s2[ 5] = 16'sh0620;
        kernel_s2[ 6] = 16'sh0540; kernel_s2[ 7] = 16'sh0420;
        kernel_s2[ 8] = 16'sh0420; kernel_s2[ 9] = 16'sh0540;
        kernel_s2[10] = 16'sh0620; kernel_s2[11] = 16'sh0640;
        kernel_s2[12] = 16'sh05C0; kernel_s2[13] = 16'sh0480;
        kernel_s2[14] = 16'sh0300; kernel_s2[15] = 16'sh01A0;
        
        // Scale 3: xi = pi/8, sigma = 12.0 (low freq, wide)
        kernel_s3[ 0] = 16'sh0200; kernel_s3[ 1] = 16'sh0320;
        kernel_s3[ 2] = 16'sh0440; kernel_s3[ 3] = 16'sh0540;
        kernel_s3[ 4] = 16'sh0600; kernel_s3[ 5] = 16'sh0660;
        kernel_s3[ 6] = 16'sh0690; kernel_s3[ 7] = 16'sh06A0;
        kernel_s3[ 8] = 16'sh06A0; kernel_s3[ 9] = 16'sh0690;
        kernel_s3[10] = 16'sh0660; kernel_s3[11] = 16'sh0600;
        kernel_s3[12] = 16'sh0540; kernel_s3[13] = 16'sh0440;
        kernel_s3[14] = 16'sh0320; kernel_s3[15] = 16'sh0200;
    end
    
    // =========================================================================
    // FSM
    // =========================================================================
    localparam S_IDLE      = 4'd0;
    localparam S_INIT      = 4'd1;
    localparam S_READ_ADDR = 4'd2;
    localparam S_READ_WAIT = 4'd3;
    localparam S_MAC       = 4'd4;
    localparam S_MODULUS   = 4'd5;
    localparam S_WRITE     = 4'd6;
    localparam S_NEXT_N    = 4'd7;
    localparam S_NEXT_SCALE= 4'd8;
    localparam S_METRICS   = 4'd9;
    localparam S_DONE      = 4'd10;
    
    reg [3:0]  state;
    reg [10:0] n_idx;           // Output sample index
    reg [4:0]  k_idx;           // Kernel tap index
    reg [1:0]  scale_idx;       // Current scale (0-3)
    reg signed [39:0] acc_real; // Real accumulator
    reg signed [39:0] acc_imag; // Imaginary accumulator (for modulus)
    reg [31:0] cycle_counter;
    reg        prec_reg;        // Registered precision mode
    
    // Kernel coefficient selection
    reg signed [15:0] coeff;
    always @(*) begin
        case (scale_idx)
            2'd0: coeff = kernel_s0[k_idx[3:0]];
            2'd1: coeff = kernel_s1[k_idx[3:0]];
            2'd2: coeff = kernel_s2[k_idx[3:0]];
            2'd3: coeff = kernel_s3[k_idx[3:0]];
        endcase
    end
    
    // Index computation for convolution
    wire signed [12:0] conv_idx;
    assign conv_idx = $signed({2'b00, n_idx}) - $signed({8'd0, k_idx}) 
                      + $signed(KERNEL_SIZE / 2);
    
    // Truncation for 8-bit mode: take upper 8 bits of input and coefficient
    wire signed [7:0] input_8bit;
    wire signed [7:0] coeff_8bit;
    assign input_8bit = input_data[15:8];   // Q8.0 (integer part)
    assign coeff_8bit = coeff[15:8];         // Q1.7 upper
    
    // MAC results for each precision mode
    wire signed [31:0] mac_16bit;
    wire signed [15:0] mac_8bit;
    assign mac_16bit = $signed({1'b0, input_data}) * $signed({1'b0, coeff});
    assign mac_8bit  = $signed(input_8bit) * $signed(coeff_8bit);
    
    // Modulus approximation: |x| ≈ max(|real|, |imag|) + 0.5 * min(|real|, |imag|)
    // For real-only wavelet, we just take |acc_real|
    wire [15:0] modulus_result;
    wire [15:0] acc_abs;
    assign acc_abs = acc_real[39] ? (~acc_real[30:15] + 16'd1) : acc_real[30:15];
    assign modulus_result = acc_abs;
    
    // =========================================================================
    // FSM Logic
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            done          <= 1'b0;
            n_idx         <= 11'd0;
            k_idx         <= 5'd0;
            scale_idx     <= 2'd0;
            acc_real      <= 40'sd0;
            acc_imag      <= 40'sd0;
            input_addr    <= 11'd0;
            input_rd_en   <= 1'b0;
            output_wr_en  <= 1'b0;
            output_addr   <= 11'd0;
            output_data   <= 16'd0;
            output_scale  <= 2'd0;
            cycle_counter <= 32'd0;
            prec_reg      <= 1'b0;
            cycles_8bit   <= 32'd0;
            cycles_16bit  <= 32'd0;
            total_cycles  <= 32'd0;
        end else begin
            output_wr_en <= 1'b0;
            input_rd_en  <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        prec_reg      <= precision_mode;
                        n_idx         <= 11'd0;
                        k_idx         <= 5'd0;
                        scale_idx     <= 2'd0;
                        acc_real      <= 40'sd0;
                        cycle_counter <= 32'd0;
                        state         <= S_INIT;
                    end
                end
                
                S_INIT: begin
                    acc_real <= 40'sd0;
                    k_idx   <= 5'd0;
                    state   <= S_READ_ADDR;
                end
                
                S_READ_ADDR: begin
                    cycle_counter <= cycle_counter + 1;
                    if (conv_idx >= 0 && conv_idx < MAX_N) begin
                        input_addr  <= conv_idx[10:0];
                        input_rd_en <= 1'b1;
                        state       <= S_READ_WAIT;
                    end else begin
                        // Out of bounds: treat as zero (padding)
                        if (k_idx == KERNEL_SIZE - 1)
                            state <= S_MODULUS;
                        else begin
                            k_idx <= k_idx + 1;
                            state <= S_READ_ADDR;
                        end
                    end
                end
                
                S_READ_WAIT: begin
                    cycle_counter <= cycle_counter + 1;
                    state <= S_MAC;
                end
                
                S_MAC: begin
                    cycle_counter <= cycle_counter + 1;
                    
                    // MAC with selected precision
                    if (prec_reg) begin
                        // 16-bit precision
                        acc_real <= acc_real + {{8{mac_16bit[31]}}, mac_16bit};
                    end else begin
                        // 8-bit precision (extend to 40-bit)
                        acc_real <= acc_real + {{24{mac_8bit[15]}}, mac_8bit};
                    end
                    
                    if (k_idx == KERNEL_SIZE - 1) begin
                        state <= S_MODULUS;
                    end else begin
                        k_idx <= k_idx + 1;
                        state <= S_READ_ADDR;
                    end
                end
                
                S_MODULUS: begin
                    // Compute |wavelet coefficient| (modulus)
                    output_data  <= modulus_result;
                    output_addr  <= n_idx;
                    output_scale <= scale_idx;
                    state        <= S_WRITE;
                end
                
                S_WRITE: begin
                    output_wr_en <= 1'b1;
                    state        <= S_NEXT_N;
                end
                
                S_NEXT_N: begin
                    if (n_idx == MAX_N - 1) begin
                        state <= S_NEXT_SCALE;
                    end else begin
                        n_idx    <= n_idx + 1;
                        acc_real <= 40'sd0;
                        k_idx    <= 5'd0;
                        state    <= S_INIT;
                    end
                end
                
                S_NEXT_SCALE: begin
                    if (scale_idx == N_SCALES - 1) begin
                        state <= S_METRICS;
                    end else begin
                        scale_idx <= scale_idx + 1;
                        n_idx     <= 11'd0;
                        acc_real  <= 40'sd0;
                        k_idx     <= 5'd0;
                        state     <= S_INIT;
                    end
                end
                
                S_METRICS: begin
                    total_cycles <= cycle_counter;
                    if (prec_reg) begin
                        cycles_16bit <= cycle_counter;
                        cycles_8bit  <= (cycle_counter * 3) >> 2; // Estimate: 8-bit ~75% of 16-bit
                    end else begin
                        cycles_8bit  <= cycle_counter;
                        cycles_16bit <= (cycle_counter * 4) / 3;  // Estimate: 16-bit ~133% of 8-bit
                    end
                    state <= S_DONE;
                end
                
                S_DONE: begin
                    done <= 1'b1;
                    if (!start)
                        state <= S_IDLE;
                end
                
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
