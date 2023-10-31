
import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--type', type=str, required=True)
opts = parser.parse_args()

def weights_string_for(path):
    return '.WEIGHTS({WEIGHTS, "' + path + '"})'

if opts.type == 'po2_multiply':
    # for po2_dot_product
    for i in range(16):
        print(f"po2_multiply #(.W(W), .I(4)) m{i} (")
        print(f"    .clk(clk), .rst(rst), .inp(a[{i}]),")
        print(f"    .zero_weight(zero_weights[{i}]), .negative_weight(negative_weights[{i}]), .log_2_weight(log_2_weights[{i}]),")
        print(f"    .result(result[{i}]), .result_v(result_v[{i}])")
        print(f");")


elif opts.type == 'po2_dot_product':
    # for po2_row_by_matrix_multiply

    def emit(i, indent):
        pad = " " * indent
        weights = weights_string_for(f"/c{i:02d}")
        print(f"{pad}po2_dot_product #(.W(W), .D(IN_D), {weights}) col{i} (")
        print(f"{pad}    .clk(clk), .rst(rst), .packed_a(packed_a), .out(dp_N_out[{i}]), .out_v(col_v[{i}])")
        print(f"{pad});")

    for i in range(4):
        emit(i, indent=4)

    print("    generate")
    print("        if (OUT_D > 4) begin")
    for i in range(4, 8):
        emit(i, indent=12)
    print("        end")
    print("    endgenerate")

    print("    generate")
    print("        if (OUT_D > 8) begin")
    for i in range(8, 16):
        emit(i, indent=12)
    print("        end")
    print("    endgenerate")
