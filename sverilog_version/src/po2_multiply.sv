
module po2_multiply #(
    parameter W = 16,       // width for each element
    parameter I = 4         // integer bits in W
)(
    input                       clk,
    input                       rst,
    input      signed [W-1:0]   inp,
    input                       negative_weight,  // 1 if weight is negative, 0 otherwise
    input             [W-1:0]   log_2_weight,     // absolute value of weight
    output reg signed [2*W-1:0] result,
    output reg                  result_v
);

    localparam
        NEGATE_1            = 0,
        NEGATE_2            = 1,
        PAD_TO_DOUBLE_WIDTH = 2,
        SHIFT               = 3,
        DONE                = 4;
    reg [3:0] state;

    // padding "constants" for zero padding single width input to double width
    reg [I-1:0] LEFT_PAD = 0;
    reg [(W-I)-1:0] RIGHT_PAD = 0;

    // temp storage for (possibly) having to negate the input
    reg [I+I-1:0] negated_integer_part;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            result_v <= 0;
            state <= negative_weight ? NEGATE_1 : PAD_TO_DOUBLE_WIDTH;
        end else
            case(state)
                NEGATE_1: begin
                    // take integer part of input
                    // and do twos compliment negation ( i.e. invert bits and add 1 )
                    negated_integer_part <= ~inp[W-1:W-I] + 1;
                    state <= NEGATE_2;
                end
                NEGATE_2: begin
                    // add negated integer part back to fractional part
                    // and convert to double width
                    // TODO: THIS IGNORES OVERFLOW with the prior +1!!! we should work in double width from start
                    result <= {LEFT_PAD, negated_integer_part, inp[W-I-1:0], RIGHT_PAD};
                    state <= SHIFT;
                end
                PAD_TO_DOUBLE_WIDTH: begin
                    // convert input to double width
                    result <= { LEFT_PAD, inp, RIGHT_PAD };
                    state <= SHIFT;
                end
                SHIFT: begin
                    // "multiply" result by the weight, by doing right shift by log2
                    result <= result >>> log_2_weight;
                    result_v <= 1;
                    state <= DONE;
                end
                DONE: begin
                end
            endcase
    end

endmodule