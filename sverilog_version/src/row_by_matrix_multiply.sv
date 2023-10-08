`default_nettype none

module row_by_matrix_multiply #(
    parameter W=16,
    parameter B_VALUES="test_matrix"
)(
  input                        clk,
  input                        rst,
  input signed [W-1:0]         a [0:3],
  output reg signed [2*W-1:0]  out [0:3],
  output reg                   out_v
);

    reg col0_v;
    reg col1_v;
    reg col2_v;
    reg col3_v;

    dot_product #(.B_VALUES({B_VALUES,"/c0.hex"})) col0 (
        .clk(clk), .rst(rst),
        .a(a), .out(out[0]), .out_v(col0_v)
    );

    dot_product #(.B_VALUES({B_VALUES,"/c1.hex"})) col1 (
        .clk(clk), .rst(rst),
        .a(a), .out(out[1]), .out_v(col1_v)
    );

    dot_product #(.B_VALUES({B_VALUES,"/c2.hex"})) col2 (
        .clk(clk), .rst(rst),
        .a(a), .out(out[2]), .out_v(col2_v)
    );

    dot_product #(.B_VALUES({B_VALUES,"/c3.hex"})) col3 (
        .clk(clk), .rst(rst),
        .a(a), .out(out[3]), .out_v(col3_v)
    );

    assign out_v = col0_v && col1_v && col2_v && col3_v;

endmodule
