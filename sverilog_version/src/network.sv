`default_nettype none

module network #(
    parameter W = 16  // width for each element
)(
    input                     clk,
    input                     rst,
    input signed [W-1:0]      inp,
    output reg signed [W-1:0] out [0:3],
    output reg                out_v
);

    localparam
        S0 = 3'b000,
        S1 = 3'b001,
        S2 = 3'b010,
        S3 = 3'b011,
        S4 = 3'b100;

    reg [2:0] net_state = S0;

    reg lsb_clk =0;
    reg signed [W-1:0] lsb_out [0:3];
    left_shift_buffer lsb (
        .clk(lsb_clk), .rst(rst),
        .inp(inp), .out(lsb_out)
    );

    reg c0_rst;
    reg signed [W-1:0] c0a0 [0:3];
    reg signed [W-1:0] c0a1 [0:3];
    reg signed [W-1:0] c0a2 [0:3];
    reg signed [W-1:0] c0a3 [0:3];
    reg signed [W-1:0] c0_out [0:3];
    reg c0_out_v;

    conv1d #(.B_VALUES("qconv0_weights")) conv0 (
        .clk(clk), .rst(c0_rst), .apply_relu(1'b1),
        .a0(c0a0), .a1(c0a1), .a2(c0a2), .a3(c0a3),
        .out(c0_out), .out_v(c0_out_v));

    reg ac_c0_clk =0;
    reg signed [W-1:0] ac_c0_0_out [0:3];
    reg signed [W-1:0] ac_c0_1_out [0:3];
    reg signed [W-1:0] ac_c0_2_out [0:3];
    reg signed [W-1:0] ac_c0_3_out [0:3];
    activation_cache activation_cache_c0_0 (
        .clk(ac_c0_clk), .rst(rst), .inp(conv0.out[0]), .out(ac_c0_0_out)
    );
    activation_cache activation_cache_c0_1 (
        .clk(ac_c0_clk), .rst(rst), .inp(conv0.out[1]), .out(ac_c0_1_out)
    );
    activation_cache activation_cache_c0_2 (
        .clk(ac_c0_clk), .rst(rst), .inp(conv0.out[2]), .out(ac_c0_2_out)
    );
    activation_cache activation_cache_c0_3 (
        .clk(ac_c0_clk), .rst(rst), .inp(conv0.out[3]), .out(ac_c0_3_out)
    );

    integer i;
    initial begin
        for(i=0; i<4; i=i+1) begin
            lsb_out[i] <= 0;
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            net_state <= S0;
        end else
            case(net_state)

                S0: begin
                    lsb_clk <= 1;
                    net_state <= S1;
                    out_v <= 0;
                end

                S1: begin
                    lsb_clk <= 0;

                    c0a0[0] <= lsb.out[0];
                    c0a0[1] <= 0;
                    c0a0[2] <= 0;
                    c0a0[3] <= 0;

                    c0a1[0] <= lsb.out[1];
                    c0a1[1] <= 0;
                    c0a1[2] <= 0;
                    c0a1[3] <= 0;

                    c0a2[0] <= lsb.out[2];
                    c0a2[1] <= 0;
                    c0a2[2] <= 0;
                    c0a2[3] <= 0;

                    c0a3[0] <= lsb.out[3];
                    c0a3[1] <= 0;
                    c0a3[2] <= 0;
                    c0a3[3] <= 0;

                    c0_rst <= 1;
                    net_state <= S2;
                end

                S2: begin
                    // wait until conv0 has run
                    c0_rst <= 0;
                    net_state <= c0_out_v ? S3 : S2;
                end

                S3: begin
                    // signal activation_cache0 to collect a value
                    ac_c0_clk <= 1;
                    net_state = S4;
                end

                S4: begin
                    ac_c0_clk <= 0;
                    out[0] <= activation_cache_c0_0.out[3];
                    out[1] <= activation_cache_c0_1.out[3];
                    out[2] <= activation_cache_c0_2.out[3];
                    out[3] <= activation_cache_c0_3.out[3];
                    out_v <= 1;
                    net_state <= S0;
                end


            endcase

    end

endmodule

