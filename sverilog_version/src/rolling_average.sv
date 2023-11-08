`default_nettype none

module rolling_average #(
    parameter W = 16,        // width for each element
    parameter LEN            // rolling average length, must be power of 2
)(
    input                     clk,
    input                     rst,
    input      signed [W-1:0] inp,
    output reg signed [W-1:0] out
);

// how big does our index into buffer need to be?
localparam MEMORY_W = $clog2(LEN)-1;

// how much do we need to scale each element before adding to buffer?
localparam SHIFT = $clog2(LEN);

reg [MEMORY_W:0] read_write_head = 0;
reg signed [W-1:0] buffer [LEN];
reg signed [2*W-1:0] rolling_total;
reg signed [W-1:0] last_evicted_value;
reg signed [W-1:0] last_added_value;

// don't start until we get at least one valid input
reg valid_in;

integer i;
initial begin
    for(i=0; i<LEN; i=i+1)
        buffer[i] <= 0;
    rolling_total <= 0;
    last_evicted_value <= 0;
    last_added_value <= 0;
    valid_in <= 0;
end

always @(posedge clk) begin
    if (valid_in == 0) begin
        if (inp > 0) valid_in = 1;
    end else begin
        read_write_head <= read_write_head + 1;
        last_evicted_value <= buffer[read_write_head];
        last_added_value <= inp;
        buffer[read_write_head] <= last_added_value;
        rolling_total <= rolling_total + last_added_value - last_evicted_value;
        out <= rolling_total >> SHIFT;
    end
end

endmodule