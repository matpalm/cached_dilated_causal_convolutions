`default_nettype none

// a . b_values -> out
// (8,) . (8,) -> (1,)

module dot_product #(
    parameter W=16,   // width for each element
    parameter D=8,    // size of packed port arrays
    parameter B_VALUES="test_b_values.hex"
)(
  input                        clk,
  input                        rst,
  input signed [D*W-1:0]       packed_a,
  output reg signed [2*W-1:0]  out,
  output reg                   out_v
);

    // unpack a
    logic signed [W-1:0] a[D];
    genvar i;
    generate
        for (i = 0; i < D; i++) begin
            assign a[i] = packed_a[W*(D-i-1) +: W];
        end
    endgenerate

    // state machine for pipelined mulitplies ( in pairs )
    // and accumulation.
    localparam
        MULT_D0     = 0,
        MULT_D1     = 1,
        MULT_D2     = 2,
        MULT_D3     = 3,
        MULT_D4     = 4,
        MULT_D5     = 5,
        MULT_D6     = 6,
        MULT_D7     = 7,
        FINAL_ADD   = 8,
        DONE        = 9;
    reg [3:0] dp_state = MULT_D0;

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
    reg signed [W-1:0] b_values [0:D-1];

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            dp_state <= MULT_D0;
            out_v <= 0;
        end else
            case(dp_state)
                // TODO: with a an array too we can fold all these
                // into one "state" with an i value; a[i] * b[i]
                MULT_D0: begin
                    acc0 <= 0;
                    product0 <= a[0] * b_values[0];
                    dp_state <= MULT_D1;
                end
                MULT_D1: begin
                    acc0 <= acc0 + product0;
                    product0 <= a[1] * b_values[1];
                    dp_state <= MULT_D2;
                end
                MULT_D2: begin
                    acc0 <= acc0 + product0;
                    product0 <= a[2] * b_values[2];
                    dp_state <= MULT_D3;
                end
                MULT_D3: begin
                    acc0 <= acc0 + product0;
                    product0 <= a[3] * b_values[3];
                    dp_state <= MULT_D4;
                    // TODO could shortcut here for 4D; dp_state <= D==4 ? FINAL_ADD : MULT_D4;
                end
                MULT_D4: begin
                    acc0 <= acc0 + product0;
                    product0 <= a[4] * b_values[4];
                    dp_state <= MULT_D5;
                end
                MULT_D5: begin
                    acc0 <= acc0 + product0;
                    product0 <= a[5] * b_values[5];
                    dp_state <= MULT_D6;
                end
                MULT_D6: begin
                    acc0 <= acc0 + product0;
                    product0 <= a[6] * b_values[6];
                    dp_state <= MULT_D7;
                end
                MULT_D7: begin
                    acc0 <= acc0 + product0;
                    product0 <= a[7] * b_values[7];
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
