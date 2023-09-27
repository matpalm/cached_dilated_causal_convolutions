
class FxpMathConv1D(object):

    def __init__(self, fxp_util): #, weights, biases):
        self.fxp_util = fxp_util

    def dot_product(self, x, weights, accumulator):
        assert len(x.shape) == 1
        assert len(weights.shape) == 1
        assert len(x) == len(weights)
        # this loop represents what could be in the state machine
        # but can be pipelined
        for i in range(len(x)):
            x_i = self.fxp_util.single_width_fxp(x[i])
            w_i = self.fxp_util.single_width_fxp(weights[i])
            prod = x_i * w_i  # will be double width
            accumulator += prod
            # keep accumulator double width. by dft a+b => +1 for int part
            self.fxp_util.resize_double_width(accumulator)
        return accumulator

    def row_by_matrix_multiply(self, x, weights, accumulators):
        assert len(x.shape) == 1
        in_d = x.shape[0]
        assert len(weights.shape) == 2
        out_d = weights.shape[0]
        assert weights.shape[1] == in_d
        assert len(accumulators) == out_d
        # this loop represents what could be in the state machine
        # but can be pipelined
        for column in range(out_d):
            accumulators[column] = self.dot_product(
                x, weights[column], accumulators[column])
        return accumulators

