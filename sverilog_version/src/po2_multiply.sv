
module po2_multiply #(
    parameter W = 16,       // width for each element
    parameter I = 4         // integer bits in W
)(
    input                     clk,
    input                     rst,
    input      signed [W-1:0] inp,
    input                     negative_weight,  // 1 if weight is negative, 0 otherwise
    input             [W-1:0] log_2_weight,     // absolute value of weight
    output reg signed [W-1:0] result,
    output reg                result_v
);

    localparam
        NEGATE_1 = 0,
        NEGATE_2 = 1,
        SHIFT    = 2,
        DONE     = 3;
    reg [2:0] state = DONE;

    reg [I-1:0] negated_integer_part;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            result_v <= 0;
            if (negative_weight) begin
                state <= NEGATE_1;
            end else begin
                result <= inp;  // need to do this since jumping ahead to SHIFT
                state <= SHIFT;
            end
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
                    // WARNING IGNORES OVERFLOW!
                    result <= {negated_integer_part, inp[W-I-1:0]};
                    state <= SHIFT;
                end
                SHIFT: begin
                    // "multiply" result by the weight, by doing right shift by log2
                    result <= result >>> log_2_weight;
                    state <= DONE;
                end
                DONE: begin
                    result_v <= 1;
                end
            endcase
    end

endmodule