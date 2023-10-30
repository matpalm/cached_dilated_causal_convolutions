import numpy as np
import os
from .util import ensure_dir_exists

class FxpMathConv1DQuantisedBitsBlock(object):

    def __init__(self, fxp_util, layer_name, weights, biases, verbose):
        self.fxp = fxp_util
        self.layer_name = layer_name
        self.verbose = verbose
        self.K = 4

        self.fxp.check_all_qIF(weights)
        self.fxp.check_all_qIF(biases)

        # weights from qkeras are [kernel, in_d, out_d] but we want
        # to slice first [kernel] then [out_d] so transpose now to
        # [kernel][out_d][in_d] to make slicing easier to read
        weights = weights.transpose(0, 2, 1)

        assert len(weights.shape) == 3
        assert weights.shape[0] == self.K
        self.out_d = weights.shape[1]
        self.in_d = weights.shape[2]

        assert len(biases.shape) == 1
        assert len(biases) == self.out_d

        self.weights = weights
        self.biases = biases


        # keep count of stats of under/overflows w.r.t double to single precision
        # conversion. these are OK, but too many means something wrong
        self._num_underflows = 0
        self._num_overflows = 0

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


    def apply(self, x):

        def to_hex(v):
            bin_str = str(v.bin())
            assert len(bin_str) % 4 == 0
            hex_str = f"{int(bin_str, 2):x}"
            target_padded_width = len(bin_str) // 4
            padding = "0" * (target_padded_width - len(hex_str))
            return padding + hex_str

        assert x.shape == (self.K, self.in_d)

        # TODO: this has diverged a bit from how the actual verilog version
        #       works but its probably not a problem..

        # prepare initial accumulators for each kernels and biases
        # note: this might be 4 accumulators ( for dilated conv ) or might
        # just be 1 accum ( for the 1x1s )
        accums = []
        for _ in range(self.K):
            accums.append([self.fxp.double_width(0) for _ in range(self.out_d)])
        double_width_biases = [self.fxp.double_width(b) for b in self.biases]

        if self.verbose:
            print("row_by_matrix_multiply inputs")
            for i in range(self.K):
                print(f"cNaI i={i} {x[i].shape} {x[i]}")

        # run each kernel; can be in parallel
        for i in range(self.K):
            self.row_by_matrix_multiply(x[i], self.weights[i], accums[i])

        if self.verbose:
            print("KERNEL OUTPUTS")
            for i in range(self.K):
                print(f"kernel i={i} {list(map(to_hex, accums[i]))}")

        # sum accumulators ( all into accum0 )
        for i in range(self.K):
            if i != 0:
                self.fxp.vector_add(accums[0], accums[i])

        # add biases
        self.fxp.vector_add(accums[0], double_width_biases)

        # resize down from double to single for output
        for i in range(self.out_d):
            self.fxp.resize_single_width(accums[0][i])

        # check for example under/overflow
        # NOTE: this are handled with clip on the double width values in verilog version
        #
        for a in accums[0]:
            if a < -7.99:
                self._num_underflows += 1
            elif a > 7.99:
                self._num_overflows += 1

        # return as np array,
        return np.array(accums[0])


    def num_underflows(self):
        return self._num_underflows

    def num_overflows(self):
        return self._num_overflows

    def export_weights_for_verilog(self, root_dir):
        # export weights for this conv1d in format
        # for loading in verilog with $readmemh

        root_dir = root_dir + "/" + self.layer_name

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

        assert len(self.weights.shape) == 3
        num_k, out_d, in_d = self.weights.shape

        for k in range(num_k):
            d = f"{root_dir}/k{k}"
            ensure_dir_exists(d)
            for o in range(out_d):
                with open(f"{d}/c{o:02d}.hex", 'w') as f:
                    for i in range(in_d):
                        f.write(single_width_hex_representation(self.weights[k, o, i]))
                        f.write(f" // {self.weights[k, o, i]}\n")

        with open(f"{root_dir}/bias.hex", 'w') as f:
            for o in range(out_d):
                f.write(double_width_hex_representation(self.biases[o]))
                f.write(f" // {self.biases[o]}\n")

    def __str__(self):
        return f"weights={self.weights.shape} biases={self.biases.shape}"
