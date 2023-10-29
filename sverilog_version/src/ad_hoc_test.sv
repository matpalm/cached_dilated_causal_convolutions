
module ad_hoc_test #(
    parameter W = 16,        // width for each element
    parameter I = 4
)(
    input                       clk,
    input                       rst,
    input      signed [W-1:0]   inp1,
    input      signed [W-1:0]   inp2,
    output reg signed [2*W-1:0] out1,
    output reg signed [2*W-1:0] out2
);

    // lower and upper bounds for single precision
    // value, in double format for comparison and single
    // precision
    // localparam int signed lower_bound_d = 32'b11111000000000000000000000000000;  // -8
    // localparam int signed upper_bound_d = 32'b00000111111111111111000000000000;  // 8

    reg [I-1:0] LEFT_PAD = 0;
    reg [(W-I)-1:0] RIGHT_PAD = 0;

    always @(posedge clk) begin
        out1 <= { LEFT_PAD, inp1, RIGHT_PAD };
        out2 <= 0;
    end

endmodule