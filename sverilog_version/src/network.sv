`default_nettype none

module network #(
    parameter W = 16  // width for each element
)(
    input rst,
    input clk,
    input sample_clk,
    input signed [W-1:0] sample_in0,
    input signed [W-1:0] sample_in1,
    input signed [W-1:0] sample_in2,
    input signed [W-1:0] sample_in3,
    output signed [W-1:0] sample_out0,
    output signed [W-1:0] sample_out1,
    output signed [W-1:0] sample_out2,
    output signed [W-1:0] sample_out3,
    input [7:0] jack
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
        CLK_ACT_CACHE_2 = 4'b1001,
        RST_CONV_3      = 4'b1010,
        CONV_3_RUNNING  = 4'b1011,
        OUTPUT          = 4'b1100;

    reg [3:0] net_state = CLK_LSB;

    //--------------------------------
    // left shift buffer

    // TODO: do we need to register sample_in0?
    // TODO: introduce three more lsbs for sample_in1, in2 and in3

    reg lsb_clk =0;
    reg signed [W-1:0] lsb_out_d0;
    reg signed [W-1:0] lsb_out_d1;
    reg signed [W-1:0] lsb_out_d2;
    reg signed [W-1:0] lsb_out_d3;

    left_shift_buffer lsb (
        .clk(lsb_clk), .rst(rst),
        .inp(sample_in0),
        .out_d0(lsb_out_d0), .out_d1(lsb_out_d1), .out_d2(lsb_out_d2), .out_d3(lsb_out_d3)
    );

    //--------------------------------
    // conv 0 block
    // always connected to left shift buffer for input

    reg c0_rst;
    reg signed [W-1:0] c0a0_d0;
    reg signed [W-1:0] c0a0_d1;
    reg signed [W-1:0] c0a0_d2;
    reg signed [W-1:0] c0a0_d3;
    reg signed [W-1:0] c0a0_d4;
    reg signed [W-1:0] c0a0_d5;
    reg signed [W-1:0] c0a0_d6;
    reg signed [W-1:0] c0a0_d7;

    reg signed [W-1:0] c0a1_d0;
    reg signed [W-1:0] c0a1_d1;
    reg signed [W-1:0] c0a1_d2;
    reg signed [W-1:0] c0a1_d3;
    reg signed [W-1:0] c0a1_d4;
    reg signed [W-1:0] c0a1_d5;
    reg signed [W-1:0] c0a1_d6;
    reg signed [W-1:0] c0a1_d7;

    reg signed [W-1:0] c0a2_d0;
    reg signed [W-1:0] c0a2_d1;
    reg signed [W-1:0] c0a2_d2;
    reg signed [W-1:0] c0a2_d3;
    reg signed [W-1:0] c0a2_d4;
    reg signed [W-1:0] c0a2_d5;
    reg signed [W-1:0] c0a2_d6;
    reg signed [W-1:0] c0a2_d7;

    reg signed [W-1:0] c0a3_d0;
    reg signed [W-1:0] c0a3_d1;
    reg signed [W-1:0] c0a3_d2;
    reg signed [W-1:0] c0a3_d3;
    reg signed [W-1:0] c0a3_d4;
    reg signed [W-1:0] c0a3_d5;
    reg signed [W-1:0] c0a3_d6;
    reg signed [W-1:0] c0a3_d7;

    reg signed [W-1:0] c0_out_d0;
    reg signed [W-1:0] c0_out_d1;
    reg signed [W-1:0] c0_out_d2;
    reg signed [W-1:0] c0_out_d3;
    reg signed [W-1:0] c0_out_d4;
    reg signed [W-1:0] c0_out_d5;
    reg signed [W-1:0] c0_out_d6;
    reg signed [W-1:0] c0_out_d7;

    reg c0_out_v;

    assign c0a0_d0 = lsb_out_d0;
    assign c0a0_d1 = 0;
    assign c0a0_d2 = 0;
    assign c0a0_d3 = 0;
    assign c0a0_d4 = 0;
    assign c0a0_d5 = 0;
    assign c0a0_d6 = 0;
    assign c0a0_d7 = 0;
    assign c0a1_d0 = lsb_out_d1;
    assign c0a1_d1 = 0;
    assign c0a1_d2 = 0;
    assign c0a1_d3 = 0;
    assign c0a1_d4 = 0;
    assign c0a1_d5 = 0;
    assign c0a1_d6 = 0;
    assign c0a1_d7 = 0;
    assign c0a2_d0 = lsb_out_d2;
    assign c0a2_d1 = 0;
    assign c0a2_d2 = 0;
    assign c0a2_d3 = 0;
    assign c0a2_d4 = 0;
    assign c0a2_d5 = 0;
    assign c0a2_d6 = 0;
    assign c0a2_d7 = 0;
    assign c0a3_d0 = lsb_out_d3;
    assign c0a3_d1 = 0;
    assign c0a3_d2 = 0;
    assign c0a3_d3 = 0;
    assign c0a3_d4 = 0;
    assign c0a3_d5 = 0;
    assign c0a3_d6 = 0;
    assign c0a3_d7 = 0;

    conv1d #(.B_VALUES("weights/qconv0")) conv0 (
        .clk(clk), .rst(c0_rst), .apply_relu(1'b1),
        .a0_d0(c0a0_d0), .a0_d1(c0a0_d1), .a0_d2(c0a0_d2), .a0_d3(c0a0_d3), .a0_d4(c0a0_d4), .a0_d5(c0a0_d5), .a0_d6(c0a0_d6), .a0_d7(c0a0_d7),
        .a1_d0(c0a1_d0), .a1_d1(c0a1_d1), .a1_d2(c0a1_d2), .a1_d3(c0a1_d3), .a1_d4(c0a1_d4), .a1_d5(c0a1_d5), .a1_d6(c0a1_d6), .a1_d7(c0a1_d7),
        .a2_d0(c0a2_d0), .a2_d1(c0a2_d1), .a2_d2(c0a2_d2), .a2_d3(c0a2_d3), .a2_d4(c0a2_d4), .a2_d5(c0a2_d5), .a2_d6(c0a2_d6), .a2_d7(c0a2_d7),
        .a3_d0(c0a3_d0), .a3_d1(c0a3_d1), .a3_d2(c0a3_d2), .a3_d3(c0a3_d3), .a3_d4(c0a3_d4), .a3_d5(c0a3_d5), .a3_d6(c0a3_d6), .a3_d7(c0a3_d7),
        .out_d0(c0_out_d0), .out_d1(c0_out_d1), .out_d2(c0_out_d2), .out_d3(c0_out_d3),
        .out_d4(c0_out_d4), .out_d5(c0_out_d5), .out_d6(c0_out_d6), .out_d7(c0_out_d7),
        .out_v(c0_out_v));

    //--------------------------------
    // conv 0 activation cache

    reg ac_c0_clk = 0;
    reg signed [W-1:0] ac_c0_d0_out_l0;
    reg signed [W-1:0] ac_c0_d0_out_l1;
    reg signed [W-1:0] ac_c0_d0_out_l2;
    reg signed [W-1:0] ac_c0_d0_out_l3;

    reg signed [W-1:0] ac_c0_d1_out_l0;
    reg signed [W-1:0] ac_c0_d1_out_l1;
    reg signed [W-1:0] ac_c0_d1_out_l2;
    reg signed [W-1:0] ac_c0_d1_out_l3;

    reg signed [W-1:0] ac_c0_d2_out_l0;
    reg signed [W-1:0] ac_c0_d2_out_l1;
    reg signed [W-1:0] ac_c0_d2_out_l2;
    reg signed [W-1:0] ac_c0_d2_out_l3;

    reg signed [W-1:0] ac_c0_d3_out_l0;
    reg signed [W-1:0] ac_c0_d3_out_l1;
    reg signed [W-1:0] ac_c0_d3_out_l2;
    reg signed [W-1:0] ac_c0_d3_out_l3;

    reg signed [W-1:0] ac_c0_d4_out_l0;
    reg signed [W-1:0] ac_c0_d4_out_l1;
    reg signed [W-1:0] ac_c0_d4_out_l2;
    reg signed [W-1:0] ac_c0_d4_out_l3;

    reg signed [W-1:0] ac_c0_d5_out_l0;
    reg signed [W-1:0] ac_c0_d5_out_l1;
    reg signed [W-1:0] ac_c0_d5_out_l2;
    reg signed [W-1:0] ac_c0_d5_out_l3;

    reg signed [W-1:0] ac_c0_d6_out_l0;
    reg signed [W-1:0] ac_c0_d6_out_l1;
    reg signed [W-1:0] ac_c0_d6_out_l2;
    reg signed [W-1:0] ac_c0_d6_out_l3;

    reg signed [W-1:0] ac_c0_d7_out_l0;
    reg signed [W-1:0] ac_c0_d7_out_l1;
    reg signed [W-1:0] ac_c0_d7_out_l2;
    reg signed [W-1:0] ac_c0_d7_out_l3;

    localparam C0_DILATION = 4;

    activation_cache #(.DILATION(C0_DILATION)) activation_cache_c0_0 (
        .clk(ac_c0_clk), .rst(rst), .inp(c0_out_d0),
        .out_l0(ac_c0_d0_out_l0),
        .out_l1(ac_c0_d0_out_l1),
        .out_l2(ac_c0_d0_out_l2),
        .out_l3(ac_c0_d0_out_l3)
    );
    activation_cache #(.DILATION(C0_DILATION)) activation_cache_c0_1 (
        .clk(ac_c0_clk), .rst(rst), .inp(c0_out_d1),
        .out_l0(ac_c0_d1_out_l0),
        .out_l1(ac_c0_d1_out_l1),
        .out_l2(ac_c0_d1_out_l2),
        .out_l3(ac_c0_d1_out_l3)
    );
    activation_cache #(.DILATION(C0_DILATION)) activation_cache_c0_2 (
        .clk(ac_c0_clk), .rst(rst), .inp(c0_out_d2),
        .out_l0(ac_c0_d2_out_l0),
        .out_l1(ac_c0_d2_out_l1),
        .out_l2(ac_c0_d2_out_l2),
        .out_l3(ac_c0_d2_out_l3)
    );
    activation_cache #(.DILATION(C0_DILATION)) activation_cache_c0_3 (
        .clk(ac_c0_clk), .rst(rst), .inp(c0_out_d3),
        .out_l0(ac_c0_d3_out_l0),
        .out_l1(ac_c0_d3_out_l1),
        .out_l2(ac_c0_d3_out_l2),
        .out_l3(ac_c0_d3_out_l3)
    );
    activation_cache #(.DILATION(C0_DILATION)) activation_cache_c0_4 (
        .clk(ac_c0_clk), .rst(rst), .inp(c0_out_d4),
        .out_l0(ac_c0_d4_out_l0),
        .out_l1(ac_c0_d4_out_l1),
        .out_l2(ac_c0_d4_out_l2),
        .out_l3(ac_c0_d4_out_l3)
    );
    activation_cache #(.DILATION(C0_DILATION)) activation_cache_c0_5 (
        .clk(ac_c0_clk), .rst(rst), .inp(c0_out_d5),
        .out_l0(ac_c0_d5_out_l0),
        .out_l1(ac_c0_d5_out_l1),
        .out_l2(ac_c0_d5_out_l2),
        .out_l3(ac_c0_d5_out_l3)
    );
    activation_cache #(.DILATION(C0_DILATION)) activation_cache_c0_6 (
        .clk(ac_c0_clk), .rst(rst), .inp(c0_out_d6),
        .out_l0(ac_c0_d6_out_l0),
        .out_l1(ac_c0_d6_out_l1),
        .out_l2(ac_c0_d6_out_l2),
        .out_l3(ac_c0_d6_out_l3)
    );
    activation_cache #(.DILATION(C0_DILATION)) activation_cache_c0_7 (
        .clk(ac_c0_clk), .rst(rst), .inp(c0_out_d7),
        .out_l0(ac_c0_d7_out_l0),
        .out_l1(ac_c0_d7_out_l1),
        .out_l2(ac_c0_d7_out_l2),
        .out_l3(ac_c0_d7_out_l3)
    );

    //--------------------------------
    // conv 1 block

    reg c1_rst = 0;

    reg signed [W-1:0] c1a0_d0;
    reg signed [W-1:0] c1a0_d1;
    reg signed [W-1:0] c1a0_d2;
    reg signed [W-1:0] c1a0_d3;
    reg signed [W-1:0] c1a0_d4;
    reg signed [W-1:0] c1a0_d5;
    reg signed [W-1:0] c1a0_d6;
    reg signed [W-1:0] c1a0_d7;

    reg signed [W-1:0] c1a1_d0;
    reg signed [W-1:0] c1a1_d1;
    reg signed [W-1:0] c1a1_d2;
    reg signed [W-1:0] c1a1_d3;
    reg signed [W-1:0] c1a1_d4;
    reg signed [W-1:0] c1a1_d5;
    reg signed [W-1:0] c1a1_d6;
    reg signed [W-1:0] c1a1_d7;

    reg signed [W-1:0] c1a2_d0;
    reg signed [W-1:0] c1a2_d1;
    reg signed [W-1:0] c1a2_d2;
    reg signed [W-1:0] c1a2_d3;
    reg signed [W-1:0] c1a2_d4;
    reg signed [W-1:0] c1a2_d5;
    reg signed [W-1:0] c1a2_d6;
    reg signed [W-1:0] c1a2_d7;

    reg signed [W-1:0] c1a3_d0;
    reg signed [W-1:0] c1a3_d1;
    reg signed [W-1:0] c1a3_d2;
    reg signed [W-1:0] c1a3_d3;
    reg signed [W-1:0] c1a3_d4;
    reg signed [W-1:0] c1a3_d5;
    reg signed [W-1:0] c1a3_d6;
    reg signed [W-1:0] c1a3_d7;

    reg signed [W-1:0] c1_out_d0;
    reg signed [W-1:0] c1_out_d1;
    reg signed [W-1:0] c1_out_d2;
    reg signed [W-1:0] c1_out_d3;
    reg signed [W-1:0] c1_out_d4;
    reg signed [W-1:0] c1_out_d5;
    reg signed [W-1:0] c1_out_d6;
    reg signed [W-1:0] c1_out_d7;

    reg c1_out_v;

    assign c1a0_d0 = ac_c0_d0_out_l0;
    assign c1a0_d1 = ac_c0_d1_out_l0;
    assign c1a0_d2 = ac_c0_d2_out_l0;
    assign c1a0_d3 = ac_c0_d3_out_l0;
    assign c1a0_d4 = ac_c0_d4_out_l0;
    assign c1a0_d5 = ac_c0_d5_out_l0;
    assign c1a0_d6 = ac_c0_d6_out_l0;
    assign c1a0_d7 = ac_c0_d7_out_l0;

    assign c1a1_d0 = ac_c0_d0_out_l1;
    assign c1a1_d1 = ac_c0_d1_out_l1;
    assign c1a1_d2 = ac_c0_d2_out_l1;
    assign c1a1_d3 = ac_c0_d3_out_l1;
    assign c1a1_d4 = ac_c0_d4_out_l1;
    assign c1a1_d5 = ac_c0_d5_out_l1;
    assign c1a1_d6 = ac_c0_d6_out_l1;
    assign c1a1_d7 = ac_c0_d7_out_l1;

    assign c1a2_d0 = ac_c0_d0_out_l2;
    assign c1a2_d1 = ac_c0_d1_out_l2;
    assign c1a2_d2 = ac_c0_d2_out_l2;
    assign c1a2_d3 = ac_c0_d3_out_l2;
    assign c1a2_d4 = ac_c0_d4_out_l2;
    assign c1a2_d5 = ac_c0_d5_out_l2;
    assign c1a2_d6 = ac_c0_d6_out_l2;
    assign c1a2_d7 = ac_c0_d7_out_l2;

    assign c1a3_d0 = ac_c0_d0_out_l3;
    assign c1a3_d1 = ac_c0_d1_out_l3;
    assign c1a3_d2 = ac_c0_d2_out_l3;
    assign c1a3_d3 = ac_c0_d3_out_l3;
    assign c1a3_d4 = ac_c0_d4_out_l3;
    assign c1a3_d5 = ac_c0_d5_out_l3;
    assign c1a3_d6 = ac_c0_d6_out_l3;
    assign c1a3_d7 = ac_c0_d7_out_l3;

    conv1d #(.B_VALUES("weights/qconv1")) conv1 (
        .clk(clk), .rst(c1_rst), .apply_relu(1'b1),
        .a0_d0(c1a0_d0), .a0_d1(c1a0_d1), .a0_d2(c1a0_d2), .a0_d3(c1a0_d3), .a0_d4(c1a0_d4), .a0_d5(c1a0_d5), .a0_d6(c1a0_d6), .a0_d7(c1a0_d7),
        .a1_d0(c1a1_d0), .a1_d1(c1a1_d1), .a1_d2(c1a1_d2), .a1_d3(c1a1_d3), .a1_d4(c1a1_d4), .a1_d5(c1a1_d5), .a1_d6(c1a1_d6), .a1_d7(c1a1_d7),
        .a2_d0(c1a2_d0), .a2_d1(c1a2_d1), .a2_d2(c1a2_d2), .a2_d3(c1a2_d3), .a2_d4(c1a2_d4), .a2_d5(c1a2_d5), .a2_d6(c1a2_d6), .a2_d7(c1a2_d7),
        .a3_d0(c1a3_d0), .a3_d1(c1a3_d1), .a3_d2(c1a3_d2), .a3_d3(c1a3_d3), .a3_d4(c1a3_d4), .a3_d5(c1a3_d5), .a3_d6(c1a3_d6), .a3_d7(c1a3_d7),
        .out_d0(c1_out_d0), .out_d1(c1_out_d1), .out_d2(c1_out_d2), .out_d3(c1_out_d3),
        .out_d4(c1_out_d4), .out_d5(c1_out_d5), .out_d6(c1_out_d6), .out_d7(c1_out_d7),
        .out_v(c1_out_v));

    //--------------------------------
    // conv 1 activation cache

    reg ac_c1_clk = 0;
    reg signed [W-1:0] ac_c1_d0_out_l0;
    reg signed [W-1:0] ac_c1_d0_out_l1;
    reg signed [W-1:0] ac_c1_d0_out_l2;
    reg signed [W-1:0] ac_c1_d0_out_l3;

    reg signed [W-1:0] ac_c1_d1_out_l0;
    reg signed [W-1:0] ac_c1_d1_out_l1;
    reg signed [W-1:0] ac_c1_d1_out_l2;
    reg signed [W-1:0] ac_c1_d1_out_l3;

    reg signed [W-1:0] ac_c1_d2_out_l0;
    reg signed [W-1:0] ac_c1_d2_out_l1;
    reg signed [W-1:0] ac_c1_d2_out_l2;
    reg signed [W-1:0] ac_c1_d2_out_l3;

    reg signed [W-1:0] ac_c1_d3_out_l0;
    reg signed [W-1:0] ac_c1_d3_out_l1;
    reg signed [W-1:0] ac_c1_d3_out_l2;
    reg signed [W-1:0] ac_c1_d3_out_l3;

    reg signed [W-1:0] ac_c1_d4_out_l0;
    reg signed [W-1:0] ac_c1_d4_out_l1;
    reg signed [W-1:0] ac_c1_d4_out_l2;
    reg signed [W-1:0] ac_c1_d4_out_l3;

    reg signed [W-1:0] ac_c1_d5_out_l0;
    reg signed [W-1:0] ac_c1_d5_out_l1;
    reg signed [W-1:0] ac_c1_d5_out_l2;
    reg signed [W-1:0] ac_c1_d5_out_l3;

    reg signed [W-1:0] ac_c1_d6_out_l0;
    reg signed [W-1:0] ac_c1_d6_out_l1;
    reg signed [W-1:0] ac_c1_d6_out_l2;
    reg signed [W-1:0] ac_c1_d6_out_l3;

    reg signed [W-1:0] ac_c1_d7_out_l0;
    reg signed [W-1:0] ac_c1_d7_out_l1;
    reg signed [W-1:0] ac_c1_d7_out_l2;
    reg signed [W-1:0] ac_c1_d7_out_l3;

    localparam C1_DILATION = 16;

    activation_cache #(.DILATION(C1_DILATION)) activation_cache_c1_0 (
        .clk(ac_c1_clk), .rst(rst), .inp(c1_out_d0),
        .out_l0(ac_c1_d0_out_l0),
        .out_l1(ac_c1_d0_out_l1),
        .out_l2(ac_c1_d0_out_l2),
        .out_l3(ac_c1_d0_out_l3)
    );
    activation_cache #(.DILATION(C1_DILATION)) activation_cache_c1_1 (
        .clk(ac_c1_clk), .rst(rst), .inp(c1_out_d1),
        .out_l0(ac_c1_d1_out_l0),
        .out_l1(ac_c1_d1_out_l1),
        .out_l2(ac_c1_d1_out_l2),
        .out_l3(ac_c1_d1_out_l3)
    );
    activation_cache #(.DILATION(C1_DILATION)) activation_cache_c1_2 (
        .clk(ac_c1_clk), .rst(rst), .inp(c1_out_d2),
        .out_l0(ac_c1_d2_out_l0),
        .out_l1(ac_c1_d2_out_l1),
        .out_l2(ac_c1_d2_out_l2),
        .out_l3(ac_c1_d2_out_l3)
    );
    activation_cache #(.DILATION(C1_DILATION)) activation_cache_c1_3 (
        .clk(ac_c1_clk), .rst(rst), .inp(c1_out_d3),
        .out_l0(ac_c1_d3_out_l0),
        .out_l1(ac_c1_d3_out_l1),
        .out_l2(ac_c1_d3_out_l2),
        .out_l3(ac_c1_d3_out_l3)
    );
    activation_cache #(.DILATION(C1_DILATION)) activation_cache_c1_4 (
        .clk(ac_c1_clk), .rst(rst), .inp(c1_out_d4),
        .out_l0(ac_c1_d4_out_l0),
        .out_l1(ac_c1_d4_out_l1),
        .out_l2(ac_c1_d4_out_l2),
        .out_l3(ac_c1_d4_out_l3)
    );
    activation_cache #(.DILATION(C1_DILATION)) activation_cache_c1_5 (
        .clk(ac_c1_clk), .rst(rst), .inp(c1_out_d5),
        .out_l0(ac_c1_d5_out_l0),
        .out_l1(ac_c1_d5_out_l1),
        .out_l2(ac_c1_d5_out_l2),
        .out_l3(ac_c1_d5_out_l3)
    );
    activation_cache #(.DILATION(C1_DILATION)) activation_cache_c1_6 (
        .clk(ac_c1_clk), .rst(rst), .inp(c1_out_d6),
        .out_l0(ac_c1_d6_out_l0),
        .out_l1(ac_c1_d6_out_l1),
        .out_l2(ac_c1_d6_out_l2),
        .out_l3(ac_c1_d6_out_l3)
    );
    activation_cache #(.DILATION(C1_DILATION)) activation_cache_c1_7 (
        .clk(ac_c1_clk), .rst(rst), .inp(c1_out_d7),
        .out_l0(ac_c1_d7_out_l0),
        .out_l1(ac_c1_d7_out_l1),
        .out_l2(ac_c1_d7_out_l2),
        .out_l3(ac_c1_d7_out_l3)
    );

    //--------------------------------
    // conv 2 block
    // always connected to conv 1 activation cache

    reg c2_rst = 0;

    reg signed [W-1:0] c2a0_d0;
    reg signed [W-1:0] c2a0_d1;
    reg signed [W-1:0] c2a0_d2;
    reg signed [W-1:0] c2a0_d3;
    reg signed [W-1:0] c2a0_d4;
    reg signed [W-1:0] c2a0_d5;
    reg signed [W-1:0] c2a0_d6;
    reg signed [W-1:0] c2a0_d7;

    reg signed [W-1:0] c2a1_d0;
    reg signed [W-1:0] c2a1_d1;
    reg signed [W-1:0] c2a1_d2;
    reg signed [W-1:0] c2a1_d3;
    reg signed [W-1:0] c2a1_d4;
    reg signed [W-1:0] c2a1_d5;
    reg signed [W-1:0] c2a1_d6;
    reg signed [W-1:0] c2a1_d7;

    reg signed [W-1:0] c2a2_d0;
    reg signed [W-1:0] c2a2_d1;
    reg signed [W-1:0] c2a2_d2;
    reg signed [W-1:0] c2a2_d3;
    reg signed [W-1:0] c2a2_d4;
    reg signed [W-1:0] c2a2_d5;
    reg signed [W-1:0] c2a2_d6;
    reg signed [W-1:0] c2a2_d7;

    reg signed [W-1:0] c2a3_d0;
    reg signed [W-1:0] c2a3_d1;
    reg signed [W-1:0] c2a3_d2;
    reg signed [W-1:0] c2a3_d3;
    reg signed [W-1:0] c2a3_d4;
    reg signed [W-1:0] c2a3_d5;
    reg signed [W-1:0] c2a3_d6;
    reg signed [W-1:0] c2a3_d7;

    reg signed [W-1:0] c2_out_d0;
    reg signed [W-1:0] c2_out_d1;
    reg signed [W-1:0] c2_out_d2;
    reg signed [W-1:0] c2_out_d3;
    reg signed [W-1:0] c2_out_d4;
    reg signed [W-1:0] c2_out_d5;
    reg signed [W-1:0] c2_out_d6;
    reg signed [W-1:0] c2_out_d7;

    reg c2_out_v;

    assign c2a0_d0 = ac_c1_d0_out_l0;
    assign c2a0_d1 = ac_c1_d1_out_l0;
    assign c2a0_d2 = ac_c1_d2_out_l0;
    assign c2a0_d3 = ac_c1_d3_out_l0;
    assign c2a0_d4 = ac_c1_d4_out_l0;
    assign c2a0_d5 = ac_c1_d5_out_l0;
    assign c2a0_d6 = ac_c1_d6_out_l0;
    assign c2a0_d7 = ac_c1_d7_out_l0;

    assign c2a1_d0 = ac_c1_d0_out_l1;
    assign c2a1_d1 = ac_c1_d1_out_l1;
    assign c2a1_d2 = ac_c1_d2_out_l1;
    assign c2a1_d3 = ac_c1_d3_out_l1;
    assign c2a1_d4 = ac_c1_d4_out_l1;
    assign c2a1_d5 = ac_c1_d5_out_l1;
    assign c2a1_d6 = ac_c1_d6_out_l1;
    assign c2a1_d7 = ac_c1_d7_out_l1;

    assign c2a2_d0 = ac_c1_d0_out_l2;
    assign c2a2_d1 = ac_c1_d1_out_l2;
    assign c2a2_d2 = ac_c1_d2_out_l2;
    assign c2a2_d3 = ac_c1_d3_out_l2;
    assign c2a2_d4 = ac_c1_d4_out_l2;
    assign c2a2_d5 = ac_c1_d5_out_l2;
    assign c2a2_d6 = ac_c1_d6_out_l2;
    assign c2a2_d7 = ac_c1_d7_out_l2;

    assign c2a3_d0 = ac_c1_d0_out_l3;
    assign c2a3_d1 = ac_c1_d1_out_l3;
    assign c2a3_d2 = ac_c1_d2_out_l3;
    assign c2a3_d3 = ac_c1_d3_out_l3;
    assign c2a3_d4 = ac_c1_d4_out_l3;
    assign c2a3_d5 = ac_c1_d5_out_l3;
    assign c2a3_d6 = ac_c1_d6_out_l3;
    assign c2a3_d7 = ac_c1_d7_out_l3;

    conv1d #(.B_VALUES("weights/qconv2")) conv2 (
        .clk(clk), .rst(c2_rst), .apply_relu(1'b1),
        .a0_d0(c2a0_d0), .a0_d1(c2a0_d1), .a0_d2(c2a0_d2), .a0_d3(c2a0_d3), .a0_d4(c2a0_d4), .a0_d5(c2a0_d5), .a0_d6(c2a0_d6), .a0_d7(c2a0_d7),
        .a1_d0(c2a1_d0), .a1_d1(c2a1_d1), .a1_d2(c2a1_d2), .a1_d3(c2a1_d3), .a1_d4(c2a1_d4), .a1_d5(c2a1_d5), .a1_d6(c2a1_d6), .a1_d7(c2a1_d7),
        .a2_d0(c2a2_d0), .a2_d1(c2a2_d1), .a2_d2(c2a2_d2), .a2_d3(c2a2_d3), .a2_d4(c2a2_d4), .a2_d5(c2a2_d5), .a2_d6(c2a2_d6), .a2_d7(c2a2_d7),
        .a3_d0(c2a3_d0), .a3_d1(c2a3_d1), .a3_d2(c2a3_d2), .a3_d3(c2a3_d3), .a3_d4(c2a3_d4), .a3_d5(c2a3_d5), .a3_d6(c2a3_d6), .a3_d7(c2a3_d7),
        .out_d0(c2_out_d0), .out_d1(c2_out_d1), .out_d2(c2_out_d2), .out_d3(c2_out_d3),
        .out_d4(c2_out_d4), .out_d5(c2_out_d5), .out_d6(c2_out_d6), .out_d7(c2_out_d7),
        .out_v(c2_out_v));

    //--------------------------------
    // conv 2 activation cache

    reg ac_c2_clk = 0;
    reg signed [W-1:0] ac_c2_d0_out_l0;
    reg signed [W-1:0] ac_c2_d0_out_l1;
    reg signed [W-1:0] ac_c2_d0_out_l2;
    reg signed [W-1:0] ac_c2_d0_out_l3;

    reg signed [W-1:0] ac_c2_d1_out_l0;
    reg signed [W-1:0] ac_c2_d1_out_l1;
    reg signed [W-1:0] ac_c2_d1_out_l2;
    reg signed [W-1:0] ac_c2_d1_out_l3;

    reg signed [W-1:0] ac_c2_d2_out_l0;
    reg signed [W-1:0] ac_c2_d2_out_l1;
    reg signed [W-1:0] ac_c2_d2_out_l2;
    reg signed [W-1:0] ac_c2_d2_out_l3;

    reg signed [W-1:0] ac_c2_d3_out_l0;
    reg signed [W-1:0] ac_c2_d3_out_l1;
    reg signed [W-1:0] ac_c2_d3_out_l2;
    reg signed [W-1:0] ac_c2_d3_out_l3;

    reg signed [W-1:0] ac_c2_d4_out_l0;
    reg signed [W-1:0] ac_c2_d4_out_l1;
    reg signed [W-1:0] ac_c2_d4_out_l2;
    reg signed [W-1:0] ac_c2_d4_out_l3;

    reg signed [W-1:0] ac_c2_d5_out_l0;
    reg signed [W-1:0] ac_c2_d5_out_l1;
    reg signed [W-1:0] ac_c2_d5_out_l2;
    reg signed [W-1:0] ac_c2_d5_out_l3;

    reg signed [W-1:0] ac_c2_d6_out_l0;
    reg signed [W-1:0] ac_c2_d6_out_l1;
    reg signed [W-1:0] ac_c2_d6_out_l2;
    reg signed [W-1:0] ac_c2_d6_out_l3;

    reg signed [W-1:0] ac_c2_d7_out_l0;
    reg signed [W-1:0] ac_c2_d7_out_l1;
    reg signed [W-1:0] ac_c2_d7_out_l2;
    reg signed [W-1:0] ac_c2_d7_out_l3;

    localparam C2_DILATION = 64;

    activation_cache #(.DILATION(C2_DILATION)) activation_cache_c2_0 (
        .clk(ac_c2_clk), .rst(rst), .inp(c2_out_d0),
        .out_l0(ac_c2_d0_out_l0),
        .out_l1(ac_c2_d0_out_l1),
        .out_l2(ac_c2_d0_out_l2),
        .out_l3(ac_c2_d0_out_l3)
    );
    activation_cache #(.DILATION(C2_DILATION)) activation_cache_c2_1 (
        .clk(ac_c2_clk), .rst(rst), .inp(c2_out_d1),
        .out_l0(ac_c2_d1_out_l0),
        .out_l1(ac_c2_d1_out_l1),
        .out_l2(ac_c2_d1_out_l2),
        .out_l3(ac_c2_d1_out_l3)
    );
    activation_cache #(.DILATION(C2_DILATION)) activation_cache_c2_2 (
        .clk(ac_c2_clk), .rst(rst), .inp(c2_out_d2),
        .out_l0(ac_c2_d2_out_l0),
        .out_l1(ac_c2_d2_out_l1),
        .out_l2(ac_c2_d2_out_l2),
        .out_l3(ac_c2_d2_out_l3)
    );
    activation_cache #(.DILATION(C2_DILATION)) activation_cache_c2_3 (
        .clk(ac_c2_clk), .rst(rst), .inp(c2_out_d3),
        .out_l0(ac_c2_d3_out_l0),
        .out_l1(ac_c2_d3_out_l1),
        .out_l2(ac_c2_d3_out_l2),
        .out_l3(ac_c2_d3_out_l3)
    );
    activation_cache #(.DILATION(C2_DILATION)) activation_cache_c2_4 (
        .clk(ac_c2_clk), .rst(rst), .inp(c2_out_d4),
        .out_l0(ac_c2_d4_out_l0),
        .out_l1(ac_c2_d4_out_l1),
        .out_l2(ac_c2_d4_out_l2),
        .out_l3(ac_c2_d4_out_l3)
    );
    activation_cache #(.DILATION(C2_DILATION)) activation_cache_c2_5 (
        .clk(ac_c2_clk), .rst(rst), .inp(c2_out_d5),
        .out_l0(ac_c2_d5_out_l0),
        .out_l1(ac_c2_d5_out_l1),
        .out_l2(ac_c2_d5_out_l2),
        .out_l3(ac_c2_d5_out_l3)
    );
    activation_cache #(.DILATION(C2_DILATION)) activation_cache_c2_6 (
        .clk(ac_c2_clk), .rst(rst), .inp(c2_out_d6),
        .out_l0(ac_c2_d6_out_l0),
        .out_l1(ac_c2_d6_out_l1),
        .out_l2(ac_c2_d6_out_l2),
        .out_l3(ac_c2_d6_out_l3)
    );
    activation_cache #(.DILATION(C2_DILATION)) activation_cache_c2_7 (
        .clk(ac_c2_clk), .rst(rst), .inp(c2_out_d7),
        .out_l0(ac_c2_d7_out_l0),
        .out_l1(ac_c2_d7_out_l1),
        .out_l2(ac_c2_d7_out_l2),
        .out_l3(ac_c2_d7_out_l3)
    );

    //---------------------------------
    // conv 3 block

    reg c3_rst = 0;

    reg signed [W-1:0] c3a0_d0;
    reg signed [W-1:0] c3a0_d1;
    reg signed [W-1:0] c3a0_d2;
    reg signed [W-1:0] c3a0_d3;
    reg signed [W-1:0] c3a0_d4;
    reg signed [W-1:0] c3a0_d5;
    reg signed [W-1:0] c3a0_d6;
    reg signed [W-1:0] c3a0_d7;

    reg signed [W-1:0] c3a1_d0;
    reg signed [W-1:0] c3a1_d1;
    reg signed [W-1:0] c3a1_d2;
    reg signed [W-1:0] c3a1_d3;
    reg signed [W-1:0] c3a1_d4;
    reg signed [W-1:0] c3a1_d5;
    reg signed [W-1:0] c3a1_d6;
    reg signed [W-1:0] c3a1_d7;

    reg signed [W-1:0] c3a2_d0;
    reg signed [W-1:0] c3a2_d1;
    reg signed [W-1:0] c3a2_d2;
    reg signed [W-1:0] c3a2_d3;
    reg signed [W-1:0] c3a2_d4;
    reg signed [W-1:0] c3a2_d5;
    reg signed [W-1:0] c3a2_d6;
    reg signed [W-1:0] c3a2_d7;

    reg signed [W-1:0] c3a3_d0;
    reg signed [W-1:0] c3a3_d1;
    reg signed [W-1:0] c3a3_d2;
    reg signed [W-1:0] c3a3_d3;
    reg signed [W-1:0] c3a3_d4;
    reg signed [W-1:0] c3a3_d5;
    reg signed [W-1:0] c3a3_d6;
    reg signed [W-1:0] c3a3_d7;

    reg signed [W-1:0] c3_out_d0;
    reg signed [W-1:0] c3_out_d1;
    reg signed [W-1:0] c3_out_d2;
    reg signed [W-1:0] c3_out_d3;
    reg signed [W-1:0] c3_out_d4;
    reg signed [W-1:0] c3_out_d5;
    reg signed [W-1:0] c3_out_d6;
    reg signed [W-1:0] c3_out_d7;

    reg c3_out_v;

    assign c3a0_d0 = ac_c2_d0_out_l0;
    assign c3a0_d1 = ac_c2_d1_out_l0;
    assign c3a0_d2 = ac_c2_d2_out_l0;
    assign c3a0_d3 = ac_c2_d3_out_l0;
    assign c3a0_d4 = ac_c2_d4_out_l0;
    assign c3a0_d5 = ac_c2_d5_out_l0;
    assign c3a0_d6 = ac_c2_d6_out_l0;
    assign c3a0_d7 = ac_c2_d7_out_l0;

    assign c3a1_d0 = ac_c2_d0_out_l1;
    assign c3a1_d1 = ac_c2_d1_out_l1;
    assign c3a1_d2 = ac_c2_d2_out_l1;
    assign c3a1_d3 = ac_c2_d3_out_l1;
    assign c3a1_d4 = ac_c2_d4_out_l1;
    assign c3a1_d5 = ac_c2_d5_out_l1;
    assign c3a1_d6 = ac_c2_d6_out_l1;
    assign c3a1_d7 = ac_c2_d7_out_l1;

    assign c3a2_d0 = ac_c2_d0_out_l2;
    assign c3a2_d1 = ac_c2_d1_out_l2;
    assign c3a2_d2 = ac_c2_d2_out_l2;
    assign c3a2_d3 = ac_c2_d3_out_l2;
    assign c3a2_d4 = ac_c2_d4_out_l2;
    assign c3a2_d5 = ac_c2_d5_out_l2;
    assign c3a2_d6 = ac_c2_d6_out_l2;
    assign c3a2_d7 = ac_c2_d7_out_l2;

    assign c3a3_d0 = ac_c2_d0_out_l3;
    assign c3a3_d1 = ac_c2_d1_out_l3;
    assign c3a3_d2 = ac_c2_d2_out_l3;
    assign c3a3_d3 = ac_c2_d3_out_l3;
    assign c3a3_d4 = ac_c2_d4_out_l3;
    assign c3a3_d5 = ac_c2_d5_out_l3;
    assign c3a3_d6 = ac_c2_d6_out_l3;
    assign c3a3_d7 = ac_c2_d7_out_l3;

    conv1d #(.B_VALUES("weights/qconv3")) conv3 (
        .clk(clk), .rst(c3_rst), .apply_relu(1'b0),
        .a0_d0(c3a0_d0), .a0_d1(c3a0_d1), .a0_d2(c3a0_d2), .a0_d3(c3a0_d3), .a0_d4(c3a0_d4), .a0_d5(c3a0_d5), .a0_d6(c3a0_d6), .a0_d7(c3a0_d7),
        .a1_d0(c3a1_d0), .a1_d1(c3a1_d1), .a1_d2(c3a1_d2), .a1_d3(c3a1_d3), .a1_d4(c3a1_d4), .a1_d5(c3a1_d5), .a1_d6(c3a1_d6), .a1_d7(c3a1_d7),
        .a2_d0(c3a2_d0), .a2_d1(c3a2_d1), .a2_d2(c3a2_d2), .a2_d3(c3a2_d3), .a2_d4(c3a2_d4), .a2_d5(c3a2_d5), .a2_d6(c3a2_d6), .a2_d7(c3a2_d7),
        .a3_d0(c3a3_d0), .a3_d1(c3a3_d1), .a3_d2(c3a3_d2), .a3_d3(c3a3_d3), .a3_d4(c3a3_d4), .a3_d5(c3a3_d5), .a3_d6(c3a3_d6), .a3_d7(c3a3_d7),
        .out_d0(c3_out_d0), .out_d1(c3_out_d1), .out_d2(c3_out_d2), .out_d3(c3_out_d3),
        .out_d4(c3_out_d4), .out_d5(c3_out_d5), .out_d6(c3_out_d6), .out_d7(c3_out_d7),
        .out_v(c3_out_v));

    //---------------------------------
    // main network state machine

    logic signed [W-1:0] out0;
    logic signed [W-1:0] out1;
    logic signed [W-1:0] out2;
    logic signed [W-1:0] out3;

    always @(posedge sample_clk) begin
        // start forward pass of network
        net_state <= CLK_LSB;
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            net_state <= CLK_LSB;
        end else
            case(net_state)

                CLK_LSB: begin
                    // signal left shift buffer to run once
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
                    // wait until conv2 has run
                    c2_rst <= 0;
                    net_state <= c2_out_v ? CLK_ACT_CACHE_2 : CONV_2_RUNNING;
                end

                CLK_ACT_CACHE_2: begin
                    // signal activation_cache 2 to collect a value
                    ac_c2_clk <= 1;
                    net_state = RST_CONV_3;
                end

                RST_CONV_3: begin
                    // signal conv3 to reset and run
                    ac_c2_clk <= 0;
                    c3_rst <= 1;
                    net_state <= CONV_3_RUNNING;
                end

                CONV_3_RUNNING: begin
                    // wait until conv3 has run
                    c3_rst <= 0;
                    net_state <= c3_out_v ? OUTPUT : CONV_3_RUNNING;
                end

                OUTPUT: begin
                    // final net output is conv2 output
                    out0 <= c3_out_d0;
                    out1 <= c3_out_d1;
                    out2 <= c3_out_d2;
                    out3 <= c3_out_d3;
                end


            endcase

    end

    assign sample_out0 = out0;
    assign sample_out1 = out1;
    assign sample_out2 = out2;
    assign sample_out3 = out3;

endmodule

