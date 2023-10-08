`default_nettype none

module row_by_matrix_multiply #(
    parameter W=16
)(
  input                        clk,
  input                        rst,
  input signed [W-1:0]         a [0:3],
  output reg signed [2*W-1:0]  out0,
  output reg signed [2*W-1:0]  out1,
  output reg signed [2*W-1:0]  out2,
  output reg signed [2*W-1:0]  out3,
  output reg                   out_v
);

    reg col0_v;
    reg col1_v;
    reg col2_v;
    reg col3_v;

    dot_product #(.B_VALUES("test_col0.hex")) col0 (
        .clk(clk), .rst(rst),
        .a(a), .out(out0), .out_v(col0_v)
    );

    dot_product #(.B_VALUES("test_col1.hex")) col1 (
        .clk(clk), .rst(rst),
        .a(a), .out(out1), .out_v(col1_v)
    );

    dot_product #(.B_VALUES("test_col2.hex")) col2 (
        .clk(clk), .rst(rst),
        .a(a), .out(out2), .out_v(col2_v)
    );

    dot_product #(.B_VALUES("test_col3.hex")) col3 (
        .clk(clk), .rst(rst),
        .a(a), .out(out3), .out_v(col3_v)
    );

    assign out_v = col0_v && col1_v && col2_v && col3_v;

endmodule
