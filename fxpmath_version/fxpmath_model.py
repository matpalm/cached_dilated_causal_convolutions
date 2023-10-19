
import pickle
import numpy as np

np.set_printoptions(precision=16)

from .fxpmath_conv1d import FxpMathConv1D
from .util import FxpUtil
from .activation_cache import ActivationCache

K = 4
VERBOSE = False

class FxpModel(object):

    def __init__(self, weights_file):

        with open(weights_file, 'rb') as f:
            self.weights = pickle.load(f)

        for weight_id in self.weights.keys():
            assert weight_id.startswith('qconv_'), weight_id

        self.num_layers = len(self.weights)
        print("|layers|", self.num_layers)

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

        self.qconvs = []
        self.activation_caches = []

        for layer_id in range(self.num_layers):
            self.qconvs.append(FxpMathConv1D(
                self.fxp,
                weights=self.weights[f"qconv_{layer_id}"]['weights'][0],
                biases=self.weights[f"qconv_{layer_id}"]['weights'][1]
                ))
            is_last_layer = layer_id == self.num_layers - 1
            if not is_last_layer:
                self.activation_caches.append(ActivationCache(
                    depth=self.in_out_d, dilation=K**(layer_id+1), kernel_size=K
                ))

    def under_and_overflow_counts(self):
        return {
            'num_underflows': sum([q.num_underflows for q in self.qconvs]),
            'num_overflows': sum([q.num_overflows for q in self.qconvs])
        }

    def predict(self, x):

        # convert to near fixed point numbers and back to floats
        x = self.fxp.nparray_to_fixed_point_floats(x)
        if VERBOSE:
            print("==============")
            print("next_x", list(x))

        # shift input values left, and add new entry to idx -1
        for i in range(K-1):
            self.input[i] = self.input[i+1]
        self.input[K-1] = x
        if VERBOSE: print("lsb", self.input)

        y_pred = self.input

        for layer_id in range(self.num_layers):
            if VERBOSE: print("layer_id", layer_id)
            is_last_layer = layer_id == self.num_layers - 1
            if not is_last_layer:
                y_pred = self.qconvs[layer_id].apply(y_pred, relu=True)
                if VERBOSE: print("post qconv y_pred", list(y_pred))
                self.activation_caches[layer_id].add(y_pred)
                y_pred = self.activation_caches[layer_id].cached_dilated_values()
                if VERBOSE: print("post activation_cache y_pred", list(y_pred))
            else:
                y_pred = self.qconvs[layer_id].apply(y_pred, relu=False)
                if VERBOSE: print("post (last) qconv y_pred", list(y_pred))

        if VERBOSE: print("y_pred", list(y_pred))
        return y_pred


