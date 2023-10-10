`default_nettype none

module conv1d #(
    parameter W=16,
    parameter B_VALUES="qconv0_weights"
)(
  input                        clk,
  input                        rst,
  input                        apply_relu,
  input signed [W-1:0]         a0_d0,
  input signed [W-1:0]         a0_d1,
  input signed [W-1:0]         a0_d2,
  input signed [W-1:0]         a0_d3,
  input signed [W-1:0]         a1_d0,
  input signed [W-1:0]         a1_d1,
  input signed [W-1:0]         a1_d2,
  input signed [W-1:0]         a1_d3,
  input signed [W-1:0]         a2_d0,
  input signed [W-1:0]         a2_d1,
  input signed [W-1:0]         a2_d2,
  input signed [W-1:0]         a2_d3,
  input signed [W-1:0]         a3_d0,
  input signed [W-1:0]         a3_d1,
  input signed [W-1:0]         a3_d2,
  input signed [W-1:0]         a3_d3,
  output reg signed [W-1:0]    out_d0,
  output reg signed [W-1:0]    out_d1,
  output reg signed [W-1:0]    out_d2,
  output reg signed [W-1:0]    out_d3,
  output reg                   out_v
);

    localparam
        MAT_MUL_RUNNING  = 3'b000,
        ACCUMULATE       = 3'b001,
        BIAS_ADD         = 3'b010,
        SINGLE_W         = 3'b011,
        OUTPUT           = 3'b100;
    reg [2:0] c1d_state = MAT_MUL_RUNNING;

    reg kernel0_v;
    reg kernel1_v;
    reg kernel2_v;
    reg kernel3_v;

    // for whatever reason these don't have a valid value (just xxx ) during accumulation
    // but _can_ access kernel0.out (?)
    reg signed [2*W-1:0]  kernel0_out [0:3];
    reg signed [2*W-1:0]  kernel1_out [0:3];
    reg signed [2*W-1:0]  kernel2_out [0:3];
    reg signed [2*W-1:0]  kernel3_out [0:3];

    // double width accumulator
    reg signed [2*W-1:0]  accum [0:3];

    // single width final result
    reg signed [W-1:0]  result [0:3];

    // bias values
    initial begin
        $readmemh({B_VALUES,"/bias.hex"}, bias_values);
    end
    reg signed [2*W-1:0] bias_values [0:3];

    // 4 kernel mat muls

    row_by_matrix_multiply #(.B_VALUES({B_VALUES,"/k0"})) kernel0 (
        .clk(clk), .rst(rst),
        .a_d0(a0_d0), .a_d1(a0_d1), .a_d2(a0_d2), .a_d3(a0_d3),
        .out_d0(kernel0_out[0]), .out_d1(kernel0_out[1]), .out_d2(kernel0_out[2]), .out_d3(kernel0_out[3]),
        .out_v(kernel0_v)
    );

    row_by_matrix_multiply #(.B_VALUES({B_VALUES,"/k1"})) kernel1 (
        .clk(clk), .rst(rst),
        .a_d0(a1_d0), .a_d1(a1_d1), .a_d2(a1_d2), .a_d3(a1_d3),
        .out_d0(kernel1_out[0]), .out_d1(kernel1_out[1]), .out_d2(kernel1_out[2]), .out_d3(kernel1_out[3]),
        .out_v(kernel1_v)
    );

    row_by_matrix_multiply #(.B_VALUES({B_VALUES,"/k2"})) kernel2 (
        .clk(clk), .rst(rst),
        .a_d0(a2_d0), .a_d1(a2_d1), .a_d2(a2_d2), .a_d3(a2_d3),
        .out_d0(kernel2_out[0]), .out_d1(kernel2_out[1]), .out_d2(kernel2_out[2]), .out_d3(kernel2_out[3]),
        .out_v(kernel2_v)
    );

    row_by_matrix_multiply #(.B_VALUES({B_VALUES,"/k3"})) kernel3 (
        .clk(clk), .rst(rst),
        .a_d0(a3_d0), .a_d1(a3_d1), .a_d2(a3_d2), .a_d3(a3_d3),
        .out_d0(kernel3_out[0]), .out_d1(kernel3_out[1]), .out_d2(kernel3_out[2]), .out_d3(kernel3_out[3]),
        .out_v(kernel3_v)
    );

    `define relu(a) (a[W-1] == 1 ) ? 0 : a

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            c1d_state <= MAT_MUL_RUNNING;
            out_v <= 0;
        end else
            case(c1d_state)
                MAT_MUL_RUNNING: begin
                    if (kernel0_v && kernel1_v && kernel2_v && kernel3_v) c1d_state = ACCUMULATE;
                end
                ACCUMULATE: begin
                    accum[0] <= kernel0_out[0] + kernel1_out[0] + kernel2_out[0] + kernel3_out[0];
                    accum[1] <= kernel0_out[1] + kernel1_out[1] + kernel2_out[1] + kernel3_out[1];
                    accum[2] <= kernel0_out[2] + kernel1_out[2] + kernel2_out[2] + kernel3_out[2];
                    accum[3] <= kernel0_out[3] + kernel1_out[3] + kernel2_out[3] + kernel3_out[3];
                    c1d_state <= BIAS_ADD;
                end
                BIAS_ADD: begin
                    accum[0] <= accum[0] + bias_values[0];
                    accum[1] <= accum[1] + bias_values[1];
                    accum[2] <= accum[2] + bias_values[2];
                    accum[3] <= accum[3] + bias_values[3];
                    c1d_state <= SINGLE_W;
                end
                SINGLE_W: begin
                    result[0] <= accum[0][27:12];
                    result[1] <= accum[1][27:12];
                    result[2] <= accum[2][27:12];
                    result[3] <= accum[3][27:12];
                    c1d_state = OUTPUT;
                end
                OUTPUT: begin
                    out_d0 <= apply_relu ? `relu(result[0]) : result[0];
                    out_d1 <= apply_relu ? `relu(result[1]) : result[1];
                    out_d2 <= apply_relu ? `relu(result[2]) : result[2];
                    out_d3 <= apply_relu ? `relu(result[3]) : result[3];
                    out_v <= 1;
                end
            endcase
    end

endmodule

