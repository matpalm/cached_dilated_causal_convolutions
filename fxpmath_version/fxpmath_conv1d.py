import numpy as np
import os

class FxpMathConv1D(object):

    def __init__(self, fxp_util, weights, biases):
        self.fxp = fxp_util

        self.fxp.check_all_qIF(weights)
        self.fxp.check_all_qIF(biases)

        # weights from qkeras are [kernel, in_d, out_d] but we want
        # to slice first [kernel] then [out_d] so transpose now to
        # [kernel][out_d][in_d] to make slicing easier to read
        weights = weights.transpose(0, 2, 1)

        assert len(weights.shape) == 3
        assert weights.shape[0] == 4  # K
        self.out_d = weights.shape[1]
        self.in_d = weights.shape[2]

        assert len(biases.shape) == 1
        assert len(biases) == self.out_d

        self.weights = weights
        self.biases = biases

        # keep count of stats of under/overflows w.r.t double to single precision
        # conversion. these are OK, but too many means something wrong
        self.num_underflows = 0
        self.num_overflows = 0

    def dot_product(self, x, weights, accumulator):
        # this loop represents what could be in the state machine
        # but can be pipelined
        for i in range(self.in_d):
            x_i = self.fxp.single_width(x[i])
            w_i = self.fxp.single_width(weights[i])
            prod = x_i * w_i  # will be double width
            accumulator.set_val(accumulator + prod)
            # keep accumulator double width. by dft a+b => +1 for int part
            self.fxp.resize_double_width(accumulator)


    def row_by_matrix_multiply(self, x, weights, accumulators):
        # this loop represents what could be in the state machine
        # but can be pipelined
        for column in range(self.out_d):
            self.dot_product(x, weights[column], accumulators[column])


    def apply(self, x, relu):

        assert len(x.shape) == 2
        assert x.shape[0] == 4  # K
        assert x.shape[1] == self.in_d

        # prepare initial accumulators for each kernsl and biases
        accum0 = [self.fxp.double_width(0) for _ in range(self.out_d)]
        accum1 = [self.fxp.double_width(0) for _ in range(self.out_d)]
        accum2 = [self.fxp.double_width(0) for _ in range(self.out_d)]
        accum3 = [self.fxp.double_width(0) for _ in range(self.out_d)]
        double_width_biases = [self.fxp.double_width(b) for b in self.biases]

        # step 1; run each kernel; can be in parallel
        self.row_by_matrix_multiply(x[0], self.weights[0], accum0)
        self.row_by_matrix_multiply(x[1], self.weights[1], accum1)
        self.row_by_matrix_multiply(x[2], self.weights[2], accum2)
        self.row_by_matrix_multiply(x[3], self.weights[3], accum3)

        # def to_hex(v):
        #     bin_str = str(v.bin())
        #     assert len(bin_str) % 4 == 0
        #     hex_str = f"{int(bin_str, 2):x}"
        #     target_padded_width = len(bin_str) // 4
        #     padding = "0" * (target_padded_width - len(hex_str))
        #     return padding + hex_str

        # print("KERNEL OUTPUTS")
        # print("kernel_0 ", list(map(to_hex, accum0)))
        # print("kernel_1 ", list(map(to_hex, accum1)))
        # print("kernel_2 ", list(map(to_hex, accum2)))
        # print("kernel_3 ", list(map(to_hex, accum3)))

        # step 2; hierarchical add, 1 of 2
        # TODO: is overflow a concern here? or is the double width
        # enough.
        self.fxp.vector_add(accum0, accum1)  # 0+1 -> 0
        self.fxp.vector_add(accum2, accum3)  # 2+3 -> 2

        # step 3; hierarchical add, 2 of 2
        self.fxp.vector_add(accum0, accum2)  # 0+2 -> 0

        # step 4; add biases
        self.fxp.vector_add(accum0, double_width_biases)

        # step 5; resize down from double to single for output
        # print("double width result h ", list(map(to_hex, accum0)))
        # print("double width result d ", list(np.array(accum0)))
        for i in range(self.out_d):
            self.fxp.resize_single_width(accum0[i])
        # print("single width result h ", list(map(to_hex, accum0)))
        # print("single width result d ", list(np.array(accum0)))

        # check for example under/overflow
        for a in accum0:
            if a < -7.99:
                self.num_underflows += 1
            elif a > 7.99:
                self.num_overflows += 1

        # step 6; apply relu, if configured
        if relu:
            for i in range(self.out_d):
                if accum0[i] < 0:
                    accum0[i] = self.fxp.double_width(0)

        # return as np array,
        return np.array(accum0)


    def export_weights_for_verilog(self, root_dir):
        # export weights for this conv1d in format
        # for loading in verilog with $readmemh

        def single_width_hex_representation(w):
            w_fp = self.fxp.single_width(w)
            if w != float(w_fp):
                raise Exception(f"??? value {k},{o},{i} ({w}) failed FP double check")
            hex_string_without_0x = w_fp.hex()[2:]
            assert len(hex_string_without_0x) == 4
            return hex_string_without_0x

        def double_width_hex_representation(w):
            w_fp = self.fxp.double_width(w)
            if w != float(w_fp):
                raise Exception(f"??? value {k},{o},{i} ({w}) failed FP double check")
            hex_string_without_0x = w_fp.hex()[2:]
            assert len(hex_string_without_0x) == 8
            return hex_string_without_0x

        def ensure_dir_exists(d):
            if not os.path.exists(d):
                os.makedirs(d)

        assert len(self.weights.shape) == 3
        num_k, out_d, in_d = self.weights.shape

        for k in range(num_k):
            d = f"{root_dir}/k{k}"
            ensure_dir_exists(d)
            for o in range(out_d):
                with open(f"{d}/c{o}.hex", 'w') as f:
                    for i in range(in_d):
                        f.write(single_width_hex_representation(self.weights[k, o, i]))
                        f.write(f" // {self.weights[k, o, i]}\n")

        with open(f"{root_dir}/bias.hex", 'w') as f:
            for o in range(out_d):
                f.write(double_width_hex_representation(self.biases[o]))
                f.write(f" // {self.biases[o]}\n")


