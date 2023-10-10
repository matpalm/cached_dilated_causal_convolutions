`default_nettype none

module row_by_matrix_multiply #(
    parameter W=16,
    parameter B_VALUES="test_matrix"
)(
  input                        clk,
  input                        rst,
  input signed [W-1:0]         a_d0,
  input signed [W-1:0]         a_d1,
  input signed [W-1:0]         a_d2,
  input signed [W-1:0]         a_d3,
  output reg signed [2*W-1:0]  out_d0,
  output reg signed [2*W-1:0]  out_d1,
  output reg signed [2*W-1:0]  out_d2,
  output reg signed [2*W-1:0]  out_d3,
  output reg                   out_v
);

    reg col0_v;
    reg col1_v;
    reg col2_v;
    reg col3_v;

    dot_product #(.B_VALUES({B_VALUES,"/c0.hex"})) col0 (
        .clk(clk), .rst(rst),
        .a_d0(a_d0), .a_d1(a_d1), .a_d2(a_d2), .a_d3(a_d3),
        .out(out_d0), .out_v(col0_v)
    );

    dot_product #(.B_VALUES({B_VALUES,"/c1.hex"})) col1 (
        .clk(clk), .rst(rst),
        .a_d0(a_d0), .a_d1(a_d1), .a_d2(a_d2), .a_d3(a_d3),
        .out(out_d1), .out_v(col1_v)
    );

    dot_product #(.B_VALUES({B_VALUES,"/c2.hex"})) col2 (
        .clk(clk), .rst(rst),
        .a_d0(a_d0), .a_d1(a_d1), .a_d2(a_d2), .a_d3(a_d3),
        .out(out_d2), .out_v(col2_v)
    );

    dot_product #(.B_VALUES({B_VALUES,"/c3.hex"})) col3 (
        .clk(clk), .rst(rst),
        .a_d0(a_d0), .a_d1(a_d1), .a_d2(a_d2), .a_d3(a_d3),
        .out(out_d3), .out_v(col3_v)
    );

    assign out_v = col0_v && col1_v && col2_v && col3_v;


endmodule
