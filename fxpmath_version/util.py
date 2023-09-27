from fxpmath import Fxp

# add vector b to entries in a



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

    def check_all_qIF(self, a):
        for v in a.flatten():
            q_val = float(self.single_width(v))
            if v != q_val:
                raise Exception(f"value {v} not representable in QI.F; it converted to {q_val}")

    def bits(self, v):
        return v.bin(frac_dot=True)

    def vector_add(self, a, b):
        assert len(a) == len(b)
        for i in range(len(a)):
            # can be parallel
            a[i].set_val(a[i] + b[i])
            self.resize_double_width(a[i])

# fxp = FxpUtil()
# a = fxp.single_width(3)
# b = fxp.single_width(4)
# ab = a + b
# fxp.resize_single_width(ab)
# print("a  ", fxp.bits(a))
# print("b  ", fxp.bits(b))
# print("ab ", fxp.bits(ab))


