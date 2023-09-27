from fxpmath import Fxp

# add vector b to entries in a
def vector_add(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
        # can be parallel
        a[i] += b[i]

class FxpUtil(object):

    def __init__(self, n_word=16, n_int=4, n_frac=12):
        self.n_word = n_word
        self.n_int = n_int
        self.n_frac = n_frac

    def single_width_fxp(self, v):
        # convert a value to the target fixed point representation for
        # values or weights
        return Fxp(v, signed=True, n_word=self.n_word, n_frac=self.n_frac)

    def double_width_fxp(self, v):
        # convert a value to the double width target fixed point
        # representation that will be used for products and accumulators
        return Fxp(v, signed=True, n_word=self.n_word*2, n_frac=self.n_frac*2)

    def resize_single_width(self, v):
        v.resize(signed=True, n_word=self.n_word, n_frac=self.n_frac)

    def resize_double_width(self, v):
        v.resize(signed=True, n_word=self.n_word*2, n_frac=self.n_frac*2)