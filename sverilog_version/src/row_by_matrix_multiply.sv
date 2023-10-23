`default_nettype none

// a . b_values -> out
// (1, 8) . (8, 8) -> (1, 8)

// TODO: could pack out as well?

module row_by_matrix_multiply #(
    parameter W=16,  // width for each element
    parameter D=16,  // size of packed port arrays
    parameter B_VALUES="test_matrix"
)(
  input                          clk,
  input                          rst,
  input signed [D*W-1:0]         packed_a,
  output reg signed [2*D*W-1:0]  packed_out,
  output reg                     out_v
);

    reg col_v [0:D-1];

    // dot product output unpacked. this variable only introduced to
    // allow a generate block for assign since it uses j in the slicing
    logic signed [2*W-1:0]  dp_N_out [0:D-1];
    genvar j;
    generate
        for (j=0; j<D; j++) begin
            localparam a = (D-j)*2*W-1;
            localparam b = (D-j-1)*2*W;
            assign packed_out[a:b] = dp_N_out[j];
        end
    endgenerate

    dot_product #(.B_VALUES({B_VALUES,"/c0.hex"})) col0 (
        .clk(clk), .rst(rst),  .packed_a(packed_a), .out(dp_N_out[0]), .out_v(col_v[0])
    );

    dot_product #(.B_VALUES({B_VALUES,"/c1.hex"})) col1 (
        .clk(clk), .rst(rst), .packed_a(packed_a), .out(dp_N_out[1]), .out_v(col_v[1])
    );

    dot_product #(.B_VALUES({B_VALUES,"/c2.hex"})) col2 (
        .clk(clk), .rst(rst), .packed_a(packed_a), .out(dp_N_out[2]), .out_v(col_v[2])
    );

    dot_product #(.B_VALUES({B_VALUES,"/c3.hex"})) col3 (
        .clk(clk), .rst(rst), .packed_a(packed_a), .out(dp_N_out[3]), .out_v(col_v[3])
    );

    dot_product #(.B_VALUES({B_VALUES,"/c4.hex"})) col4 (
        .clk(clk), .rst(rst), .packed_a(packed_a), .out(dp_N_out[4]), .out_v(col_v[4])
    );

    dot_product #(.B_VALUES({B_VALUES,"/c5.hex"})) col5 (
        .clk(clk), .rst(rst), .packed_a(packed_a), .out(dp_N_out[5]), .out_v(col_v[5])
    );

    dot_product #(.B_VALUES({B_VALUES,"/c6.hex"})) col6 (
        .clk(clk), .rst(rst), .packed_a(packed_a), .out(dp_N_out[6]), .out_v(col_v[6])
    );

    dot_product #(.B_VALUES({B_VALUES,"/c7.hex"})) col7 (
        .clk(clk), .rst(rst), .packed_a(packed_a), .out(dp_N_out[7]), .out_v(col_v[7])
    );

    // TODO: is it enough to just check one?
    assign out_v = col_v[0] && col_v[1] && col_v[2] && col_v[3] && col_v[4] && col_v[5] && col_v[6] && col_v[7];


endmodule
