from fxpmath import Fxp
import numpy as np
import os

class FxpUtil(object):

    def __init__(self, n_word=16, n_int=4, n_frac=12):
        self.n_word = n_word
        self.n_int = n_int
        self.n_frac = n_frac

    def single_width(self, v):
        # convert a value to the target fixed point representation for
        # values or weights
        return Fxp(v, signed=True, n_word=self.n_word, n_frac=self.n_frac)

    def double_width(self, v):
        # convert a value to the double width target fixed point
        # representation that will be used for products and accumulators
        return Fxp(v, signed=True, n_word=self.n_word*2, n_frac=self.n_frac*2)

    def resize_single_width(self, v):
        v.resize(signed=True, n_word=self.n_word, n_frac=self.n_frac)

    def resize_double_width(self, v):
        v.resize(signed=True, n_word=self.n_word*2, n_frac=self.n_frac*2)

    def check_all_log2(self, a):
        it = np.nditer(np.abs(a), flags=['multi_index'])
        for v in it:
            log2_v = np.log2(v)
            if int(log2_v) != log2_v:
                raise Exception(f"value {v} [{it.multi_index}] not negative power of 2"
                                f" log2_v={log2_v}")

    def check_all_qIF(self, a):
        it = np.nditer(a, flags=['multi_index'])
        for v in it:
            q_val = float(self.single_width(v))
            if v != q_val:
                raise Exception(f"value {v} [{it.multi_index}] not representable in"
                                f" QI.F; it converted to {q_val}")

    def bits(self, v):
        return v.bin(frac_dot=True)

    def vector_add(self, a, b):
        assert len(a) == len(b)
        for i in range(len(a)):
            # can be parallel
            a[i].set_val(a[i] + b[i])
            self.resize_double_width(a[i])

    # util to convert numpy array X to float values in QI.F
    def nparray_to_fixed_point_floats(self, a):
        def cast_to_fp_and_back(v):
            return float(self.single_width(v))
        return np.vectorize(cast_to_fp_and_back)(a)

    def _bit_not(self, n):
        return (1 << self.n_int) - 1 - n

    def _twos_comp_to_signed(self, n):
        if (1 << (self.n_int-1) & n) > 0:
            return -int(self._bit_not(n) + 1)
        else:
            return int(n)

    def fixed_point_to_decimal(self, fixed_point_binary):
        # TODO! assumes FP 4/12. #lazy
        assert self.n_int == 4
        assert self.n_frac == 12
        integer_bits = fixed_point_binary >> 12
        integer_value = self._twos_comp_to_signed(integer_bits)
        fractional_bits = fixed_point_binary & 0xFFF
        fractional_value = fractional_bits / float(2**12)
        return integer_value + fractional_value


# fxp = FxpUtil()
# a = fxp.single_width(3)
# b = fxp.single_width(4)
# ab = a + b
# fxp.resize_single_width(ab)
# print("a  ", fxp.bits(a))
# print("b  ", fxp.bits(b))
# print("ab ", fxp.bits(ab))

def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)