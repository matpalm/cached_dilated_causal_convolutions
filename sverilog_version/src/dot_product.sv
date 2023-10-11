`default_nettype none

module dot_product #(
    parameter W=16,
    parameter B_VALUES="test_b_values.hex"
)(
  input                        clk,
  input                        rst,
  input signed [W-1:0]         a_d0,
  input signed [W-1:0]         a_d1,
  input signed [W-1:0]         a_d2,
  input signed [W-1:0]         a_d3,
  output reg signed [2*W-1:0]  out,
  output reg                   out_v
);

    // state machine for pipelined mulitplies ( in pairs )
    // and accumulation.
    localparam
        MULT_D0     = 3'b000,
        MULT_D1     = 3'b001,
        MULT_D2     = 3'b010,
        MULT_D3     = 3'b011,
        FINAL_ADD   = 3'b100,
        DONE        = 3'b101;
    reg [2:0] dp_state = MULT_D0;

    // see https://projectf.io/posts/fixed-point-numbers-in-verilog/
    reg signed [2*W-1:0] acc0;
    reg signed [2*W-1:0] product0;
    //reg signed [2*W-1:0] acc1;
    //reg signed [2*W-1:0] product1;

    // b values for dot product are network weights and are
    // provided by B_VALUES module level param
    initial begin
        $readmemh(B_VALUES, b_values);
        out_v <= 0;
    end
    reg signed [W-1:0] b_values [0:3];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            dp_state <= MULT_D0;
            out_v <= 0;
        end else
            case(dp_state)
                MULT_D0: begin
                    acc0 <= 0;
                    product0 <= a_d0 * b_values[0];
                    dp_state <= MULT_D1;
                end
                MULT_D1: begin
                    acc0 <= acc0 + product0;
                    product0 <= a_d1 * b_values[1];
                    dp_state <= MULT_D2;
                end
                MULT_D2: begin
                    acc0 <= acc0 + product0;
                    product0 <= a_d2 * b_values[2];
                    dp_state <= MULT_D3;
                end
                MULT_D3: begin
                    acc0 <= acc0 + product0;
                    product0 <= a_d3 * b_values[3];
                    dp_state <= FINAL_ADD;
                end
                FINAL_ADD: begin
                    acc0 <= acc0 + product0;
                    dp_state <= DONE;
                end
                DONE: begin
                    out <= acc0;
                    out_v <= 1;
                end
            endcase

    end

endmodule
