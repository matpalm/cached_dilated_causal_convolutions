`default_nettype none

module left_shift_buffer #(
    parameter W = 16  // width for each element
)(
    input              clk,
    input              rst,
    input [W-1:0]      inp,
    output reg [W-1:0] out [0:3]
);

    reg [W-1:0] buffer [0:3];
    integer i;

    generate
        genvar j;
        for(j=0; j<4; j=j+1)
            assign out[j] = buffer[j];
    endgenerate

    initial begin
        for(i=0; i<4; i=i+1)
            buffer[i] <= 0;
    end

    always @(posedge clk) begin

        buffer[0] <= inp;

        for(i=1; i<4; i=i+1)
            buffer[i] <= buffer[i-1];

    end

endmodule