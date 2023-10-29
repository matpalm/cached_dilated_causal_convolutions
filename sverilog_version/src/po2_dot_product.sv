`default_nettype none

// a . b_values -> out

module po2_dot_product #(
    parameter W=16,     // width for each element
    parameter D,        // size of packed port arrays  TODO: generalise later
    parameter WEIGHTS   // root dir for weight hex files
)(
  input                        clk,
  input                        rst,
  input signed [D*W-1:0]       packed_a,
  output reg signed [2*W-1:0]  out,
  output reg                   out_v
);

    // unpack a
    logic signed [W-1:0] a[D];
    genvar j;
    generate
        for (j=0; j<D; j++) begin
            assign a[j] = packed_a[W*(D-j-1) +: W];
        end
    endgenerate

    localparam
        MULTIPLYING_ELEMENTS  = 0,
        ADD_1                 = 1,
        ADD_2                 = 2,
        DONE                  = 3;
    reg [2:0] state;

    // D bits representing whether the Dth weight is zero or negative
    reg zero_weights [0:D-1];
    reg negative_weights [0:D-1];

    // D weights representing the log values of the weights
    reg [W-1:0] log_2_weights [0:D-1];

    // b values for dot product are network weights and are
    // provided by B_VALUES module level param
    initial begin
        $readmemh({WEIGHTS, "/zero_weights.hex"}, zero_weights);
        $readmemh({WEIGHTS, "/negative_weights.hex"}, negative_weights);
        $readmemh({WEIGHTS, "/log_2_weights.hex"}, log_2_weights);
        out_v <= 0;
    end

    // D double width values to hold the po2 multiplications
    wire signed [2*W-1:0] result [0:D-1];

    // accumulator for all po2_multiply results
    reg signed [2*W-1:0] accumulator [0:(D/2)-1];

    // D bit values for whether the Dth multiplcation is finished
    reg [D-1:0] result_v;

    po2_multiply #(.W(W), .I(4)) m0 (
        .clk(clk), .rst(rst), .inp(a[0]),
        .zero_weight(zero_weights[0]), .negative_weight(negative_weights[0]), .log_2_weight(log_2_weights[0]),
        .result(result[0]), .result_v(result_v[0])
    );
    po2_multiply #(.W(W), .I(4)) m1 (
        .clk(clk), .rst(rst), .inp(a[1]),
        .zero_weight(zero_weights[1]), .negative_weight(negative_weights[1]), .log_2_weight(log_2_weights[1]),
        .result(result[1]), .result_v(result_v[1])
    );
    po2_multiply #(.W(W), .I(4)) m2 (
        .clk(clk), .rst(rst), .inp(a[2]),
        .zero_weight(zero_weights[2]), .negative_weight(negative_weights[2]), .log_2_weight(log_2_weights[2]),
        .result(result[2]), .result_v(result_v[2])
    );
    po2_multiply #(.W(W), .I(4)) m3 (
        .clk(clk), .rst(rst), .inp(a[3]),
        .zero_weight(zero_weights[3]), .negative_weight(negative_weights[3]), .log_2_weight(log_2_weights[3]),
        .result(result[3]), .result_v(result_v[3])
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= MULTIPLYING_ELEMENTS;
            out_v <= 0;
        end else
            case(state)
                MULTIPLYING_ELEMENTS: begin
                    state <= (result_v == '1) ? ADD_1 : MULTIPLYING_ELEMENTS;
                end
                ADD_1: begin
                    accumulator[0] <= result[0] + result[1];
                    accumulator[1] <= result[2] + result[3];
                    state <= ADD_2;
                end
                ADD_2: begin
                    accumulator[0] <= accumulator[0] + accumulator[1];
                    state <= DONE;
                end
                DONE: begin
                    out <= accumulator[0];
                    out_v <= 1;
                end
            endcase

    end

endmodule
