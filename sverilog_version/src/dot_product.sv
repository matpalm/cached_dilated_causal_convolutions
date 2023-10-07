`default_nettype none

module dot_product #(
    parameter W=16,
    parameter B_VALUES="test_b_values.hex"
)(
  input                      clk,
  input                      rst,
  input signed [W-1:0]       a [0:7],
  output reg signed [W-1:0]  out,
  output reg                 out_v
);

    // state machine for pipelined mulitplies ( in pairs )
    // and accumulation.
    localparam
        MULT_01     = 3'b000,
        MULT_23     = 3'b001,
        MULT_45     = 3'b010,
        MULT_67     = 3'b011,
        FINAL_ADD_1 = 3'b100,
        FINAL_ADD_2 = 3'b101,
        DONE        = 3'b110;
    reg [2:0] dp_state = MULT_01;

    // see https://projectf.io/posts/fixed-point-numbers-in-verilog/
    logic signed [2*W-1:0] acc0;
    logic signed [2*W-1:0] product0;
    logic signed [2*W-1:0] acc1;
    logic signed [2*W-1:0] product1;

    // b values for dot product are network weights and are
    // provided by B_VALUES module level param
    initial begin
        $readmemh(B_VALUES, b_values);
        out_v <= 0;
    end
    reg signed [W-1:0] b_values [0:7];


    always @(posedge clk or posedge rst) begin
        if (rst) begin
            dp_state <= MULT_01;
            out_v <= 0;
        end else
            case(dp_state)
                MULT_01: begin
                    acc0 <= 0;
                    acc1 <= 0;
                    product0 <= a[0] * b_values[0];
                    product1 <= a[1] * b_values[1];
                    dp_state <= MULT_23;
                end
                MULT_23: begin
                    acc0 <= acc0 + product0;
                    acc1 <= acc1 + product1;
                    product0 <= a[2] * b_values[2];
                    product1 <= a[3] * b_values[3];
                    dp_state <= MULT_45;
                end
                MULT_45: begin
                    acc0 <= acc0 + product0;
                    acc1 <= acc1 + product1;
                    product0 <= a[4] * b_values[4];
                    product1 <= a[5] * b_values[5];
                    dp_state <= MULT_67;
                end
                MULT_67: begin
                    acc0 <= acc0 + product0;
                    acc1 <= acc1 + product1;
                    product0 <= a[6] * b_values[6];
                    product1 <= a[7] * b_values[7];
                    dp_state <= FINAL_ADD_1;
                end
                FINAL_ADD_1: begin
                    acc0 <= acc0 + product0;
                    acc1 <= acc1 + product1;
                    dp_state <= FINAL_ADD_2;
                end
                FINAL_ADD_2: begin
                    acc0 <= acc0 + acc1;
                    dp_state <= DONE;
                end
                DONE: begin
                    out <= acc0[27:12];  // single precision rounding
                    out_v <= 1;
                end
            endcase

    end

endmodule
