import numpy as np
import os
import math
from .util import ensure_dir_exists, nearest_log2_value_or_zero

DP_COUNT = 0

class FxpMathConv1DPO2Block(object):

    def __init__(self, fxp_util, layer_name, weights, biases, verbose):
        self.fxp = fxp_util
        self.layer_name = layer_name
        self.verbose = verbose

        # sometimes the weights from the qkeras model even with po2 might be _just_ off
        # so we do conversion now with a degree of tolerance
        weights = np.vectorize(nearest_log2_value_or_zero)(weights)

        # double check weights
        self.fxp.check_all_log2_or_zero(weights)
        self.fxp.check_all_qIF(biases)

        # BUT NOT FOR PO2
        # weights from qkeras are [kernel, in_d, out_d] but we want
        # to slice first [kernel] then [out_d] so transpose now to
        # [kernel][out_d][in_d] to make slicing easier to read
        weights = weights.transpose(0, 2, 1)

        assert len(weights.shape) == 3
        self.K = weights.shape[0]
        assert self.K in [1, 4]
        self.out_d = weights.shape[1]
        self.in_d = weights.shape[2]

        assert len(biases.shape) == 1
        assert len(biases) == self.out_d

        # check there are no weights with abs(values) over 1
        if weights.min() < -1:
            raise Exception(f"no po2 weight value should be < -1  weights.min()={weights.min()}")
        if weights.max() > 1:
            raise Exception(f"no po2 weight value should be > +1  weights.max()={weights.max()}")

        # keep track of three things about the weights;
        # 1) which are negative
        # 2) log2 of FP number
        # 3) which, after FP conversion, became zero ( even though this shouldn't be )
        self.negative_weights = np.zeros_like(weights, dtype=bool)
        self.weights_log2 = np.zeros_like(weights, dtype=int)
        self.zero_weights = np.zeros_like(weights, dtype=bool)

        for idx in np.ndindex(weights.shape):
            # do conversion to FP first since it handles weird corner cases of
            # FXP math better.
            fp_v = self.fxp.single_width(weights[idx])
            if fp_v == 0:
                self.zero_weights[idx] = True
            else:
                if fp_v < 0:
                    self.negative_weights[idx] = True
                    fp_v *= -1
                assert fp_v > 0
                weight_log2 = math.log2(fp_v)
                if weight_log2 != int(weight_log2):
                    raise Exception(f"there was a weight with a non int log 2 value {idx}")
                weight_log2 = int(-weight_log2)
                assert weight_log2 >= 0
                self.weights_log2[idx] = weight_log2

        # TODO remove once working with neg weights etc!
        self.weights = weights

        # biases are always FP trained with quantised_bits
        self.biases = biases

        # keep count of stats of under/overflows w.r.t double to single precision
        # conversion. these are OK, but too many means something wrong
        self._num_underflows = 0
        self._num_overflows = 0

    def dot_product(self, x, negative_weights, weights_log2, zero_weights, accumulator):

        global DP_COUNT

        # this loop represents what could be in the state machine
        # but can be pipelined
        for i in range(self.in_d):

            x_i = self.fxp.single_width(x[i])

            # check for zero case
            if zero_weights[i]:
                prod = self.fxp.single_width(0)
            else:
                # negate and shift
                prod = -x_i if negative_weights[i] else x_i
                prod >>= weights_log2[i]
            if self.verbose:
                print(f"dp={DP_COUNT:10d} i={i:2d} x_i={x_i} nw_i={negative_weights[i]}"
                    f" w_log2_i={weights_log2[i]} zw_i={zero_weights[i]} => prod={prod}")

            # add to accumulator
            if self.verbose:
                print(f"adding prod {prod} {self.fxp.bits(prod)} to"
                    f" accumulator {accumulator} {self.fxp.bits(accumulator)}")
            accumulator.set_val(accumulator + prod)

            # keep accumulator double width. by dft a+b => +1 for int part
            self.fxp.resize_double_width(accumulator)

            DP_COUNT += 1


    def row_by_matrix_multiply(self, x, negative_weights, weights_log2, zero_weights, accumulators):
        # this loop represents what could be in the state machine
        # but can be pipelined
        for column in range(self.out_d):
            self.dot_product(x,
                negative_weights[column], weights_log2[column], zero_weights[column],
                accumulators[column])


    def apply(self, x):

        def to_hex(v):
            bin_str = str(v.bin())
            assert len(bin_str) % 4 == 0
            hex_str = f"{int(bin_str, 2):x}"
            target_padded_width = len(bin_str) // 4
            padding = "0" * (target_padded_width - len(hex_str))
            return padding + hex_str

        if self.K == 1:
            # expect a vector (D,), so turn into a (1, D)
            assert len(x.shape) == 1
            x = x.reshape((1, -1))

        assert len(x.shape) == 2
        assert x.shape[0] == self.K
        assert x.shape[1] == self.in_d

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
            self.row_by_matrix_multiply(x[i],
                self.negative_weights[i], self.weights_log2[i], self.zero_weights[i],
                accums[i])

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
        root_dir = f"{root_dir}/{self.layer_name}"

        num_k, out_d, in_d = self.negative_weights.shape
        print("num_k", num_k, "out_d", out_d, "in_d", in_d)

        for k in range(num_k):
            d = f"{root_dir}/k{k}"
            ensure_dir_exists(d)
            for o in range(out_d):

                # write pow2 weights
                with open(f"{d}/w_l2_{o:02d}.hex", 'w') as f:
                    for i in range(in_d):
                        v = self.weights_log2[k, o, i]
                        print(hex(v)[2:], file=f)

                # TODO: decide how to write negative_weights and zero_weights bits

        print("self.negative_weights", self.negative_weights.shape)
        print("self.weights_log2", self.weights_log2.shape)
        print("self.zero_weights", self.zero_weights.shape)

    def __str__(self):
        return f" negative_weights={self.negative_weights.shape}" \
               f" weights_log2={self.weights_log2.shape}" \
               f" zero_weights={self.zero_weights.shape}"