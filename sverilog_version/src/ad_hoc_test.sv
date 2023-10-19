
module ad_hoc_test #(
    parameter W = 16        // width for each element
)(
    input                       clk,
    input                       rst,
    input      signed [2*W-1:0] inp1,
    input      signed [2*W-1:0] inp2,
    output reg signed [W-1:0]   out1,
    output reg signed [W-1:0]   out2
);

    // lower and upper bounds for single precision
    // value, in double format for comparison and single
    // precision
    localparam int signed lower_bound_d = 32'b11111000000000000000000000000000;  // -8
    localparam int signed upper_bound_d = 32'b00000111111111111111000000000000;  // 8

    always @(posedge clk) begin
        out1 <= inp1 < lower_bound_d ? 1 : 0;  // underflow
        out2 <= inp1 > upper_bound_d ? 1 : 0;  // overflow
//        out <= inp[27:12];
    end

endmodule