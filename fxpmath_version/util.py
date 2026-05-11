from fxpmath import Fxp
import numpy as np
import os
import math

class FxpUtil(object):

    def __init__(self):
        self.n_int = int(os.getenv("N_INT", 4))
        self.n_frac = int(os.getenv("N_FRAC", 12))
        self.n_word = self.n_int + self.n_frac
        # print(f"FP N_INT={self.n_int} N_FRAC={self.n_frac}")

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

    def check_all_log2_or_zero(self, a):
        it = np.nditer(np.abs(a), flags=['multi_index'])
        for v in it:
            if v != 0:
                try:
                    log2_v = math.log2(v)
                    if int(log2_v) != log2_v:
                        raise Exception(f"value {v} [{it.multi_index}] not negative power of 2"
                                        f" log2_v={log2_v}")
                except ValueError as ve:
                    raise Exception(f"value error [{ve}] from v={v}")

    def check_all_qIF(self, a, double_width: bool = False):
        it = np.nditer(a, flags=['multi_index'])
        for v in it:
            if double_width:
                q_val = float(self.double_width(v))
            else:
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
        integer_bits = fixed_point_binary >> self.n_frac
        integer_value = self._twos_comp_to_signed(integer_bits)
        fractional_bits = fixed_point_binary & (1<<self.n_frac)-1
        fractional_value = fractional_bits / float(2**self.n_frac)
        return integer_value + fractional_value

def ensure_dir_exists(d):
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except FileExistsError:
            # can happen as race condition
            pass


def ensure_dir_exists_for_file(f):
    ensure_dir_exists(os.path.dirname(os.path.abspath(f)))


def nearest_log2_value_or_zero(v, atol=1e-5):
    try:
        if v == 0:
            return 0
        negative_v = v < 0
        l2v = math.log2(abs(v))
        l2v_rounded = round(l2v)
        if abs(l2v - l2v_rounded) > atol:
            raise Exception(f"value={v} has log2_value={l2v} which rounds to {l2v_rounded}"
                            f" which is not close enough given atol={atol}")
        rv = 2 ** l2v_rounded
        return -rv if negative_v else rv
    except Exception as e:
        print(f"??? v={v} e={e}")
