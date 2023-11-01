
module po2_multiply #(
    parameter W = 16,       // width for each element
    parameter I = 4         // integer bits in W
)(
    input                       clk,
    input                       rst,
    input      signed [W-1:0]   inp,
    input                       zero_weight,      // 1 if weight is zero, 0 otherwise
    input                       negative_weight,  // 1 if weight is negative, 0 otherwise
    input             [W-1:0]   log_2_weight,     // absolute value of weight
    output reg signed [2*W-1:0] result,           // double width result
    output reg                  result_v
);

    localparam
        NEGATE              = 0,
        SHIFT               = 1,
        EMIT_ZERO           = 2,
        DONE                = 3;
    reg [1:0] state;

    // padding constants for conversion from single to double width.
    `define LEFT_PAD_0 {I{1'b0}}
    `define LEFT_PAD_1 {I{1'b1}}
    `define RIGHT_PAD {(W-I){1'b0}}

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            result_v <= 0;
            if (zero_weight | inp==0)
                // if either weight, or input, is zero we just emit zero
                state <= EMIT_ZERO;
            else begin
                // pad the input to double width. note: we need to retain sign of input
                result <= { ((inp < 0) ? `LEFT_PAD_1 : `LEFT_PAD_0), inp, `RIGHT_PAD };
                // then we either negate, or jump straight to shifting
                state <= negative_weight ? NEGATE : SHIFT;
            end

        end else
            case(state)
                NEGATE: begin
                    // twos compliment negation is usually invert bits and add 1
                    // but for fixed point we need to add 2^I, not 1.
                    result <= ~result + (1 << I);
                    state <= SHIFT;
                end
                SHIFT: begin
                    // "multiply" result by the weight, by doing right shift by log2 ( retaining sign )
                    result <= result >>> log_2_weight;
                    result_v <= 1;
                    state <= DONE;
                end
                EMIT_ZERO: begin
                    result <= 0;
                    result_v <= 1;
                    state <= DONE;
                end
                DONE: begin
                end
            endcase
    end

endmodule