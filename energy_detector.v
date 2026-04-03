// energy_detector.v — Signal Energy Detector for Adaptive Precision Control
// ============================================================================
// Computes running signal energy (sum of squared samples) over a sliding
// window and compares against a programmable threshold.
//
// Output: precision_mode
//   0 = LOW  precision (8-bit)  — calm market, save power
//   1 = HIGH precision (16-bit) — volatile market, preserve accuracy
//
// The energy is computed as: E = sum(x[i]^2) for i in window
// Uses a simple accumulator with add-new/subtract-old for sliding window.
//
// Fixed-point: input Q8.8 (16-bit signed), energy accumulator 40-bit
// ============================================================================

module energy_detector #(
    parameter WINDOW_SIZE = 64,      // Energy measurement window
    parameter THRESHOLD   = 32'h00100000  // Default threshold (tunable)
)(
    input  wire        clk,
    input  wire        rst_n,
    
    // Data input
    input  wire        data_valid,     // Pulse when new sample arrives
    input  wire signed [15:0] data_in, // Q8.8 input sample
    
    // Threshold (programmable via top-level)
    input  wire [31:0] energy_threshold,
    
    // Output
    output reg         precision_mode,  // 0=8bit, 1=16bit
    output reg  [31:0] current_energy   // For debug/monitoring
);

    // =========================================================================
    // Circular buffer for sliding window
    // =========================================================================
    reg signed [15:0] sample_buf [0:WINDOW_SIZE-1];
    reg [$clog2(WINDOW_SIZE)-1:0] buf_ptr;
    reg [$clog2(WINDOW_SIZE):0]   sample_count;
    
    // Energy accumulator (40-bit to avoid overflow with 64 x 16-bit^2 = 32-bit + margin)
    reg [39:0] energy_acc;
    
    // Squared value of new and old samples
    wire [31:0] new_sq;
    wire [31:0] old_sq;
    
    wire signed [15:0] old_sample;
    assign old_sample = sample_buf[buf_ptr];
    
    // Unsigned squares (absolute value squared)
    wire [15:0] abs_new;
    wire [15:0] abs_old;
    assign abs_new = data_in[15] ? (~data_in + 16'd1) : data_in;
    assign abs_old = old_sample[15] ? (~old_sample + 16'd1) : old_sample;
    
    assign new_sq = abs_new * abs_new;  // 16x16 = 32-bit unsigned
    assign old_sq = abs_old * abs_old;
    
    // Hysteresis threshold (10% below and above to prevent rapid switching)
    wire [31:0] threshold_high;
    wire [31:0] threshold_low;
    assign threshold_high = energy_threshold;
    assign threshold_low  = energy_threshold - (energy_threshold >> 3); // ~12.5% hysteresis
    
    // =========================================================================
    // Main logic
    // =========================================================================
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_ptr        <= 0;
            sample_count   <= 0;
            energy_acc     <= 40'd0;
            precision_mode <= 1'b0;  // start in low-precision (8-bit)
            current_energy <= 32'd0;
            
            for (i = 0; i < WINDOW_SIZE; i = i + 1) begin
                sample_buf[i] <= 16'sd0;
            end
        end else if (data_valid) begin
            // Sliding window energy update
            if (sample_count >= WINDOW_SIZE) begin
                // Window is full: subtract old, add new
                energy_acc <= energy_acc - {8'd0, old_sq} + {8'd0, new_sq};
            end else begin
                // Window filling up: just add
                energy_acc    <= energy_acc + {8'd0, new_sq};
                sample_count  <= sample_count + 1;
            end
            
            // Store new sample and advance pointer
            sample_buf[buf_ptr] <= data_in;
            buf_ptr <= (buf_ptr == WINDOW_SIZE - 1) ? 0 : buf_ptr + 1;
            
            // Update output energy (truncate to 32 bits for comparison)
            current_energy <= energy_acc[31:0];
            
            // Precision mode decision with hysteresis
            if (!precision_mode && energy_acc[31:0] > threshold_high) begin
                precision_mode <= 1'b1;  // Switch to HIGH precision (16-bit)
            end else if (precision_mode && energy_acc[31:0] < threshold_low) begin
                precision_mode <= 1'b0;  // Switch to LOW precision (8-bit)
            end
        end
    end

endmodule
