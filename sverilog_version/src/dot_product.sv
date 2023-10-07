`default_nettype none

module dot_product #(
    parameter W=16
)(
  input                      clk,
  input                      rst,
  input signed [W-1:0]       a [0:7],
  input signed [W-1:0]       b [0:7],
  output reg signed [W-1:0]  out,
  output reg                 valid_o
);

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

    // assign out = acc;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            dp_state <= MULT_01;
            valid_o <= 0;
        end else
            case(dp_state)
                MULT_01: begin
                    acc0 <= 0;
                    acc1 <= 0;
                    product0 <= a[0] * b[0];
                    product1 <= a[1] * b[1];
                    valid_o <= 0;
                    dp_state <= MULT_23;
                end
                MULT_23: begin
                    acc0 <= acc0 + product0;
                    acc1 <= acc1 + product1;
                    product0 <= a[2] * b[2];
                    product1 <= a[3] * b[3];
                    dp_state <= MULT_45;
                end
                MULT_45: begin
                    acc0 <= acc0 + product0;
                    acc1 <= acc1 + product1;
                    product0 <= a[4] * b[4];
                    product1 <= a[5] * b[5];
                    dp_state <= MULT_67;
                end
                MULT_67: begin
                    acc0 <= acc0 + product0;
                    acc1 <= acc1 + product1;
                    product0 <= a[6] * b[6];
                    product1 <= a[7] * b[7];
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
                    valid_o <= 1;
                end
            endcase

    end

endmodule
