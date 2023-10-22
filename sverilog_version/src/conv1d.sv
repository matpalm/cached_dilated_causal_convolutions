`default_nettype none

module conv1d #(
    parameter W=16,  // width for each element
    parameter D=8,   // size of packed port arrays
    parameter B_VALUES="qconv0_weights"
)(
  input                        clk,
  input                        rst,
  input                        apply_relu,
  input signed [D*W-1:0]       packed_a0,
  input signed [D*W-1:0]       packed_a1,
  input signed [D*W-1:0]       packed_a2,
  input signed [D*W-1:0]       packed_a3,
  output reg signed [D*W-1:0]  packed_out,
  output reg                   out_v
);

    localparam
        MAT_MUL_RUNNING  = 3'b000,
        ACCUMULATE       = 3'b001,
        BIAS_ADD         = 3'b010,
        CLIP_LOWER       = 3'b011,
        CLIP_UPPER       = 3'b100,
        SINGLE_W         = 3'b101,
        APPLY_RELU       = 3'b110,
        OUTPUT           = 3'b111;
    reg [2:0] state = MAT_MUL_RUNNING;

    reg kernel0_v;
    reg kernel1_v;
    reg kernel2_v;
    reg kernel3_v;

    // for whatever reason these don't have a valid value (just xxx ) during accumulation
    // but _can_ access kernel0.out (?)
    reg signed [2*W-1:0]  kernel0_out [0:7];
    reg signed [2*W-1:0]  kernel1_out [0:7];
    reg signed [2*W-1:0]  kernel2_out [0:7];
    reg signed [2*W-1:0]  kernel3_out [0:7];

    // double width accumulator
    reg signed [2*W-1:0]  accum [0:7];

    // single width final result
    reg signed [W-1:0]  result [0:7];

    // bias values
    initial begin
        $readmemh({B_VALUES,"/bias.hex"}, bias_values);
    end
    reg signed [2*W-1:0] bias_values [0:7];

    // 4 kernel mat muls

    row_by_matrix_multiply #(.B_VALUES({B_VALUES,"/k0"})) kernel0 (
        .clk(clk), .rst(rst),
        .packed_a(packed_a0),
        .out_d0(kernel0_out[0]), .out_d1(kernel0_out[1]), .out_d2(kernel0_out[2]), .out_d3(kernel0_out[3]),
        .out_d4(kernel0_out[4]), .out_d5(kernel0_out[5]), .out_d6(kernel0_out[6]), .out_d7(kernel0_out[7]),
        .out_v(kernel0_v)
    );

    row_by_matrix_multiply #(.B_VALUES({B_VALUES,"/k1"})) kernel1 (
        .clk(clk), .rst(rst),
        .packed_a(packed_a1),
        .out_d0(kernel1_out[0]), .out_d1(kernel1_out[1]), .out_d2(kernel1_out[2]), .out_d3(kernel1_out[3]),
        .out_d4(kernel1_out[4]), .out_d5(kernel1_out[5]), .out_d6(kernel1_out[6]), .out_d7(kernel1_out[7]),
        .out_v(kernel1_v)
    );

    row_by_matrix_multiply #(.B_VALUES({B_VALUES,"/k2"})) kernel2 (
        .clk(clk), .rst(rst),
        .packed_a(packed_a2),
        .out_d0(kernel2_out[0]), .out_d1(kernel2_out[1]), .out_d2(kernel2_out[2]), .out_d3(kernel2_out[3]),
        .out_d4(kernel2_out[4]), .out_d5(kernel2_out[5]), .out_d6(kernel2_out[6]), .out_d7(kernel2_out[7]),
        .out_v(kernel2_v)
    );

    row_by_matrix_multiply #(.B_VALUES({B_VALUES,"/k3"})) kernel3 (
        .clk(clk), .rst(rst),
        .packed_a(packed_a3),
        .out_d0(kernel3_out[0]), .out_d1(kernel3_out[1]), .out_d2(kernel3_out[2]), .out_d3(kernel3_out[3]),
        .out_d4(kernel3_out[4]), .out_d5(kernel3_out[5]), .out_d6(kernel3_out[6]), .out_d7(kernel3_out[7]),
        .out_v(kernel3_v)
    );

    `define relu(a) (a[W-1] == 1 ) ? 0 : a

    integer i;

    // the max value for single precision is 7.999755859375 whereas the min value is -8
    // so to avoid overflow we clip the double width precision
    // value between these bounds _before_ the single precision conversion
    localparam int signed lower_bound = 32'b11111000000000000000000000000000;  // -8
    localparam int signed upper_bound = 32'b00000111111111111111000000000000;  // 7.999755859375

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= MAT_MUL_RUNNING;
            out_v <= 0;
        end else
            case(state)
                MAT_MUL_RUNNING: begin
                    if (kernel0_v && kernel1_v && kernel2_v && kernel3_v) state = ACCUMULATE;
                end
                ACCUMULATE: begin
                    for (i=0; i<D; i=i+1) begin
                        accum[i] <= kernel0_out[i] + kernel1_out[i] + kernel2_out[i] + kernel3_out[i];
                    end
                    state <= BIAS_ADD;
                end
                BIAS_ADD: begin
                    for (i=0; i<D; i=i+1) begin
                        accum[i] <= accum[i] + bias_values[i];
                    end
                    state <= CLIP_LOWER;
                end
                CLIP_LOWER: begin
                    for (i=0; i<D; i=i+1) begin
                        accum[i] <= accum[i] < lower_bound ? lower_bound : accum[i];
                    end
                    state <= CLIP_UPPER;
                end
                CLIP_UPPER: begin
                    for (i=0; i<D; i=i+1) begin
                        accum[i] <= accum[i] > upper_bound ? upper_bound : accum[i];
                    end
                    state <= SINGLE_W;
                end
                SINGLE_W: begin
                    for (i=0; i<D; i=i+1) begin
                        result[i] <= accum[i][27:12];
                    end
                    state = APPLY_RELU;
                end
                APPLY_RELU: begin
                    for (i=0; i<D; i=i+1) begin
                        result[i] <= apply_relu ? `relu(result[i]) : result[i];
                    end
                    state = OUTPUT;
                end
                OUTPUT: begin
                    // TODO can't do this ?
                    // for (i=0; i<D; i=i+1) begin
                    //     packed_out[(D-i)*W-1:(D-i-1)*W] <= result[i];
                    // end

                    // TODO!!!! having to do this assumes D=8 :/
                    packed_out[8*W-1:7*W] <= result[0];
                    packed_out[7*W-1:6*W] <= result[1];
                    packed_out[6*W-1:5*W] <= result[2];
                    packed_out[5*W-1:4*W] <= result[3];
                    packed_out[4*W-1:3*W] <= result[4];
                    packed_out[3*W-1:2*W] <= result[5];
                    packed_out[2*W-1:1*W] <= result[6];
                    packed_out[1*W-1:0*W] <= result[7];
                    out_v <= 1;
                end
            endcase
    end

endmodule

