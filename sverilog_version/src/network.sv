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
        CLK_LSB         = 4'b0000,
        RST_CONV_0      = 4'b0001,
        CONV_0_RUNNING  = 4'b0010,
        CLK_ACT_CACHE_0 = 4'b0011,
        RST_CONV_1      = 4'b0100,
        CONV_1_RUNNING  = 4'b0101,
        CLK_ACT_CACHE_1 = 4'b0110,
        RST_CONV_2      = 4'b0111,
        CONV_2_RUNNING  = 4'b1000,
        OUTPUT          = 4'b1001;

    reg [3:0] net_state = CLK_LSB;

    //--------------------------------
    // left shift buffer

    reg lsb_clk =0;
    reg signed [W-1:0] lsb_out [0:3];
    left_shift_buffer lsb (
        .clk(lsb_clk), .rst(rst),
        .inp(inp), .out(lsb_out)
    );

    //--------------------------------
    // conv 0 block
    // always connected to left shift buffer for input

    reg c0_rst;
    reg signed [W-1:0] c0a0 [0:3];
    reg signed [W-1:0] c0a1 [0:3];
    reg signed [W-1:0] c0a2 [0:3];
    reg signed [W-1:0] c0a3 [0:3];
    reg signed [W-1:0] c0_out [0:3];
    reg c0_out_v;

    assign c0a0[0] = lsb.out[0];
    assign c0a0[1] = 0;
    assign c0a0[2] = 0;
    assign c0a0[3] = 0;
    assign c0a1[0] = lsb.out[1];
    assign c0a1[1] = 0;
    assign c0a1[2] = 0;
    assign c0a1[3] = 0;
    assign c0a2[0] = lsb.out[2];
    assign c0a2[1] = 0;
    assign c0a2[2] = 0;
    assign c0a2[3] = 0;
    assign c0a3[0] = lsb.out[3];
    assign c0a3[1] = 0;
    assign c0a3[2] = 0;
    assign c0a3[3] = 0;

    conv1d #(.B_VALUES("qconv0_weights")) conv0 (
        .clk(clk), .rst(c0_rst), .apply_relu(1'b1),
        .a0(c0a0), .a1(c0a1), .a2(c0a2), .a3(c0a3),
        .out(c0_out), .out_v(c0_out_v));

    //--------------------------------
    // conv 0 activation cache

    reg ac_c0_clk = 0;
    reg signed [W-1:0] ac_c0_0_out [0:3];
    reg signed [W-1:0] ac_c0_1_out [0:3];
    reg signed [W-1:0] ac_c0_2_out [0:3];
    reg signed [W-1:0] ac_c0_3_out [0:3];
    activation_cache #(.DILATION(4)) activation_cache_c0_0 (
        .clk(ac_c0_clk), .rst(rst), .inp(conv0.out[0]), .out(ac_c0_0_out)
    );
    activation_cache #(.DILATION(4)) activation_cache_c0_1 (
        .clk(ac_c0_clk), .rst(rst), .inp(conv0.out[1]), .out(ac_c0_1_out)
    );
    activation_cache #(.DILATION(4)) activation_cache_c0_2 (
        .clk(ac_c0_clk), .rst(rst), .inp(conv0.out[2]), .out(ac_c0_2_out)
    );
    activation_cache #(.DILATION(4)) activation_cache_c0_3 (
        .clk(ac_c0_clk), .rst(rst), .inp(conv0.out[3]), .out(ac_c0_3_out)
    );

    //--------------------------------
    // conv 1 block
    // always connected to conv 0 activation cache

    reg c1_rst = 0;
    reg signed [W-1:0] c1a0 [0:3];
    reg signed [W-1:0] c1a1 [0:3];
    reg signed [W-1:0] c1a2 [0:3];
    reg signed [W-1:0] c1a3 [0:3];
    reg signed [W-1:0] c1_out [0:3];
    reg c1_out_v;

    assign c1a0 = activation_cache_c0_0.out;
    assign c1a1 = activation_cache_c0_1.out;
    assign c1a2 = activation_cache_c0_2.out;
    assign c1a3 = activation_cache_c0_3.out;

    conv1d #(.B_VALUES("qconv1_weights")) conv1 (
        .clk(clk), .rst(c1_rst), .apply_relu(1'b1),
        .a0(c1a0), .a1(c1a1), .a2(c1a2), .a3(c1a3),
        .out(c1_out), .out_v(c1_out_v));

    //--------------------------------
    // conv 1 activation cache

    reg ac_c1_clk = 0;
    reg signed [W-1:0] ac_c1_0_out [0:3];
    reg signed [W-1:0] ac_c1_1_out [0:3];
    reg signed [W-1:0] ac_c1_2_out [0:3];
    reg signed [W-1:0] ac_c1_3_out [0:3];
    activation_cache #(.DILATION(16)) activation_cache_c1_0 (
        .clk(ac_c1_clk), .rst(rst), .inp(conv1.out[0]), .out(ac_c1_0_out)
    );
    activation_cache #(.DILATION(16)) activation_cache_c1_1 (
        .clk(ac_c1_clk), .rst(rst), .inp(conv1.out[1]), .out(ac_c1_1_out)
    );
    activation_cache #(.DILATION(16)) activation_cache_c1_2 (
        .clk(ac_c1_clk), .rst(rst), .inp(conv1.out[2]), .out(ac_c1_2_out)
    );
    activation_cache #(.DILATION(16)) activation_cache_c1_3 (
        .clk(ac_c1_clk), .rst(rst), .inp(conv1.out[3]), .out(ac_c1_3_out)
    );

    //--------------------------------
    // conv 2 block
    // always connected to conv 1 activation cache

    // TODO do we need c2a0 etc? or can we just sub in activation_cache_c1_0.out directly?
    reg c2_rst = 0;
    reg signed [W-1:0] c2a0 [0:3];
    reg signed [W-1:0] c2a1 [0:3];
    reg signed [W-1:0] c2a2 [0:3];
    reg signed [W-1:0] c2a3 [0:3];
    reg signed [W-1:0] c2_out [0:3];
    reg c2_out_v;

    assign c2a0 = activation_cache_c1_0.out;
    assign c2a1 = activation_cache_c1_1.out;
    assign c2a2 = activation_cache_c1_2.out;
    assign c2a3 = activation_cache_c1_3.out;

    conv1d #(.B_VALUES("qconv2_weights")) conv2 (
        .clk(clk), .rst(c2_rst), .apply_relu(1'b0),
        .a0(c2a0), .a1(c2a1), .a2(c2a2), .a3(c2a3),
        .out(c2_out), .out_v(c2_out_v));

    //-----------------------------
    // initialisation

    integer i;
    initial begin
        for(i=0; i<4; i=i+1) begin
            lsb_out[i] <= 0;
        end
    end

    //---------------------------------
    // main network state machine

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            net_state <= CLK_LSB;
        end else
            case(net_state)

                CLK_LSB: begin
                    // signal left shift buffer to run once
                    out_v <= 0;
                    lsb_clk <= 1;
                    net_state <= RST_CONV_0;
                end

                RST_CONV_0: begin
                    // signal conv0 to reset and run
                    lsb_clk <= 0;
                    c0_rst <= 1;
                    net_state <= CONV_0_RUNNING;
                end

                CONV_0_RUNNING: begin
                    // wait until conv0 has run
                    c0_rst <= 0;
                    net_state <= c0_out_v ? CLK_ACT_CACHE_0 : CONV_0_RUNNING;
                end

                CLK_ACT_CACHE_0: begin
                    // signal activation_cache 0 to collect a value
                    ac_c0_clk <= 1;
                    net_state = RST_CONV_1;
                end

                RST_CONV_1: begin
                    // signal conv1 to reset and run
                    ac_c0_clk <= 0;
                    c1_rst <= 1;
                    net_state <= CONV_1_RUNNING;
                end

                CONV_1_RUNNING: begin
                    // wait until conv1 has run
                    c1_rst <= 0;
                    net_state <= c1_out_v ? CLK_ACT_CACHE_1 : CONV_1_RUNNING;
                end

                CLK_ACT_CACHE_1: begin
                    // signal activation_cache 1 to collect a value
                    ac_c1_clk <= 1;
                    net_state = RST_CONV_2;
                end

                RST_CONV_2: begin
                    // signal conv2 to reset and run
                    ac_c1_clk <= 0;
                    c2_rst <= 1;
                    net_state <= CONV_2_RUNNING;
                end

                CONV_2_RUNNING: begin
                    // wait until conv1 has run
                    c2_rst <= 0;
                    net_state <= c2_out_v ? OUTPUT : CONV_2_RUNNING;
                end

                OUTPUT: begin
                    // final net output is conv2 output
                    out[0] <= conv2.out[0];
                    out[1] <= conv2.out[1];
                    out[2] <= conv2.out[2];
                    out[3] <= conv2.out[3];
                    out_v <= 1;
                    net_state <= CLK_LSB;
                end


            endcase

    end

endmodule

