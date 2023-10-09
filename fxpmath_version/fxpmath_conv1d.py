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


    def dot_product(self, x, weights, accumulator):
        # this loop represents what could be in the state machine
        # but can be pipelined
        for i in range(self.in_d):
            x_i = self.fxp.single_width(x[i])
            w_i = self.fxp.single_width(weights[i])
            print("dp", x_i, w_i)
            prod = x_i * w_i  # will be double width
            accumulator.set_val(accumulator + prod)
            # keep accumulator double width. by dft a+b => +1 for int part
            self.fxp.resize_double_width(accumulator)


    def row_by_matrix_multiply(self, x, weights, accumulators):
        # this loop represents what could be in the state machine
        # but can be pipelined
        for column in range(self.out_d):
            print(f">row_by_matrix_multiply column={column}")
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
        print("< x0 w0 -> accum0=", accum0)
        print("...", [self.fxp.bits(a) for a in accum0])

        print("x1 w1")
        self.row_by_matrix_multiply(x[1], self.weights[1], accum1)
        print("x2 w2")
        self.row_by_matrix_multiply(x[2], self.weights[2], accum2)
        print("x3 w3")
        self.row_by_matrix_multiply(x[3], self.weights[3], accum3)

        # step 2; hierarchical add, 1 of 2
        # TODO: is overflow a concern here? or is the double width
        # enough.
        self.fxp.vector_add(accum0, accum1)  # 0+1 -> 0
        self.fxp.vector_add(accum2, accum3)  # 2+3 -> 2

        # step 3; hierarchical add, 2 of 2
        self.fxp.vector_add(accum0, accum2)  # 0+2 -> 0

        # step 4; add biases
        print("add biases", double_width_biases)
        self.fxp.vector_add(accum0, double_width_biases)

        print("Result after add bias", accum0)
        print("...", [self.fxp.bits(a) for a in accum0])

        # step 5; resize down from double to single for output
        for i in range(self.out_d):
            self.fxp.resize_single_width(accum0[i])

        # step 6; apply relu, if configured
        if relu:
            for i in range(self.out_d):
                if accum0[i] < 0:
                    accum0[i] = self.fxp.double_width(0)

        # return as np array,
        return np.array(accum0)


    def export_weights_per_dot_product(self, root_dir):
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


