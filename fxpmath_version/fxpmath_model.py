
import pickle
import numpy as np

np.set_printoptions(precision=16)

from .fxpmath_conv1d import FxpMathConv1D
from .util import FxpUtil
from .activation_cache import ActivationCache

K = 4

class FxpModel(object):

    def __init__(self, weights_file):

        with open(weights_file, 'rb') as f:
            self.weights = pickle.load(f)

        # assume just two convs
        assert sorted(self.weights.keys()) == ['qconv_0', 'qconv_1', 'qconv_2', 'qconv_3']

        # use first conv to derive in/out size
        # recall; for now we assume in==out
        # and all other convs are the same sized
        self.in_out_d = None
        for key in self.weights.keys():
            weights = self.weights[key]['weights'][0]
            num_kernels, out_d, in_d = weights.shape
            assert num_kernels == 4
            assert out_d == in_d
            if self.in_out_d == None:
                self.in_out_d = in_d
            else:
                assert self.in_out_d == in_d

        # general fxp util
        self.fxp = FxpUtil()

        # buffer for layer0 input
        self.input = np.zeros((K, self.in_out_d), dtype=np.float32)

        self.qconv0 = FxpMathConv1D(
            self.fxp,
            weights=self.weights['qconv_0']['weights'][0],
            biases=self.weights['qconv_0']['weights'][1]
            )
        self.activation_cache_0 = ActivationCache(
            depth=self.in_out_d, dilation=4**1, kernel_size=4
        )

        self.qconv1 = FxpMathConv1D(
            self.fxp,
            weights=self.weights['qconv_1']['weights'][0],
            biases=self.weights['qconv_1']['weights'][1]
            )
        self.activation_cache_1 = ActivationCache(
            depth=self.in_out_d, dilation=4**2, kernel_size=4
        )

        self.qconv2 = FxpMathConv1D(
            self.fxp,
            weights=self.weights['qconv_2']['weights'][0],
            biases=self.weights['qconv_2']['weights'][1]
            )
        self.activation_cache_2 = ActivationCache(
            depth=self.in_out_d, dilation=4**3, kernel_size=4
        )

        self.qconv3 = FxpMathConv1D(
            self.fxp,
            weights=self.weights['qconv_3']['weights'][0],
            biases=self.weights['qconv_3']['weights'][1]
            )


    def predict(self, x):

        # convert to near fixed point numbers and back to floats
        x = self.fxp.nparray_to_fixed_point_floats(x)
        print("next_x", x)

        # shift input values left, and add new entry to idx -1
        for i in range(K-1):
            self.input[i] = self.input[i+1]
        self.input[K-1] = x
        print("lsb", self.input)

        # pass input to qconv0, then activation cache
        y_pred = self.qconv0.apply(self.input, relu=True)
        print("qconv0.out", y_pred)
        self.activation_cache_0.add(y_pred)
        y_pred = self.activation_cache_0.cached_dilated_values()
        print("activation_cache_0.out", y_pred)

        # pass cached values to qconv1, then next activate cache
        y_pred = self.qconv1.apply(y_pred, relu=True)
        print("qconv1.out", y_pred)
        self.activation_cache_1.add(y_pred)
        y_pred = self.activation_cache_1.cached_dilated_values()
        print("activation_cache_1.out", y_pred)

        # pass cached values to qconv2, then next activate cache
        y_pred = self.qconv2.apply(y_pred, relu=True)
        print("qconv2.out", y_pred)
        self.activation_cache_2.add(y_pred)
        y_pred = self.activation_cache_2.cached_dilated_values()
        print("activation_cache_2.out", y_pred)

        # pass cached values to qconv3, without relu, for result
        y_pred = self.qconv3.apply(y_pred, relu=False)
        print("qconv3.out", y_pred)
        return y_pred

if __name__ == '__main__':
    fxp_model = FxpModel(weights_file='qkeras_weights.pkl')
    print(fxp_model.predict([0,0,0]))