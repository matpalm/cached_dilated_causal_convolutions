`default_nettype none

// a . b_values -> out

module po2_dot_product #(
    parameter W=16,     // width for each element
    parameter D,        // size of dot product
    parameter WEIGHTS   // root dir for weight hex files
)(
  input                        clk,
  input                        rst,
  input signed [D*W-1:0]       packed_a,
  output reg signed [2*W-1:0]  out,
  output reg                   out_v
);

    // unpack a
    logic signed [W-1:0] a[D];
    genvar j;
    generate
        for (j=0; j<D; j++) begin
            assign a[j] = packed_a[W*(D-j-1) +: W];
        end
    endgenerate

    localparam
        START_NEXT_MULT  = 0,
        WAIT_FOR_MULT    = 1,
        ACCUMULATE       = 2,
        DONE             = 3;
    reg [2:0] state;

    // D bits representing whether the Dth weight is zero or negative
    reg zero_weights [0:D-1];
    reg negative_weights [0:D-1];

    // D weights representing the log values of the weights
    reg [W-1:0] log_2_weights [0:D-1];

    // b values for dot product are network weights and are
    // provided by WEIGHTS module level param
    initial begin
        $readmemh({WEIGHTS, "/zero_weights.hex"}, zero_weights);
        $readmemh({WEIGHTS, "/negative_weights.hex"}, negative_weights);
        $readmemh({WEIGHTS, "/log_2_weights.hex"}, log_2_weights);
        out_v <= 0;
    end

    // accumulator for all po2_multiply results
    reg signed [2*W-1:0] accumulator;

    // registers to handle in/out of po2 mult
    reg                  po2_rst;
    reg signed [W-1:0]   po2_inp;
    reg                  po2_zero_weight;
    reg                  po2_negative_weight;
    reg [W-1:0]          po2_log_2_weight;
    reg signed [2*W-1:0] po2_result;
    reg                  po2_result_v;

    po2_multiply #(.W(W), .I(4)) po2_mult (
        .clk(clk), .rst(po2_rst), .inp(po2_inp),
        .zero_weight(po2_zero_weight), .negative_weight(po2_negative_weight),
        .log_2_weight(po2_log_2_weight),
        .result(po2_result), .result_v(po2_result_v)
    );

    integer i;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            accumulator <= 0;
            i <= 0;
            state <= START_NEXT_MULT;
            out_v <= 0;
        end else
            case(state)
                START_NEXT_MULT: begin
                    po2_rst <= 1;
                    po2_inp <= a[i];
                    po2_zero_weight <= zero_weights[i];
                    po2_negative_weight <= negative_weights[i];
                    po2_log_2_weight <= log_2_weights[i];
                    state <= WAIT_FOR_MULT;
                end
                WAIT_FOR_MULT: begin
                    po2_rst <= 0;
                    if (po2_result_v == 1) state <= ACCUMULATE;
                end
                ACCUMULATE: begin
                    accumulator <= accumulator + po2_result;
                    i <= i + 1;
                    state <= (i == D-1) ? DONE : START_NEXT_MULT;
                end
                DONE: begin
                    out <= accumulator;
                    out_v <= 1;
                end
            endcase
    end

endmodule
