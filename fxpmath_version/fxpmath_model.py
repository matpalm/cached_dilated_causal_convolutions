
import pickle
import numpy as np

np.set_printoptions(precision=16)

from .fxpmath_conv1d_qb import FxpMathConv1DQuantisedBitsBlock
from .fxpmath_conv1d_po2 import FxpMathConv1DPO2Block
from .util import FxpUtil
from .activation_cache import ActivationCache

K = 4

class FxpModel(object):

    def __init__(self, weights_file, verbose):

        with open(weights_file, 'rb') as f:
            self.weights = pickle.load(f)

        weight_ids = list(self.weights.keys())
        print("weight_ids", weight_ids)
        self.weight_ids = weight_ids
        for weight_id in weight_ids:
            assert (weight_id.startswith('qconv_qb_') or weight_id.startswith('qconv_po2')) , weight_id

        self.num_layers = len(weight_ids)
        print("|layers|", self.num_layers)

        # scan each conv to derive in/out size
        in_ds = {}
        out_ds = {}
        for weight_id in weight_ids:
            weights = self.weights[weight_id]['weights'][0]
            num_kernels, in_d, out_d = weights.shape
            print("weight_id", weight_id, "num_kernels, in_d, out_d", num_kernels, in_d, out_d)
            in_ds[weight_id] = in_d
            out_ds[weight_id] = out_d
            if weight_id.startswith('qconv_qb_'):
                assert num_kernels == 4
            elif weight_id.startswith('qconv_po2'):
                assert num_kernels in [1, 4]

        print(f"in_ds={in_ds} & out_ds={out_ds}")
        if in_ds[weight_ids[0]] != out_ds[weight_ids[-1]]:
            raise Exception(f"expected first layer in_d to be same as last layer out_d"
                            f" but was in_ds={in_ds} out_ds={out_ds}")

        # general fxp util
        self.fxp = FxpUtil()

        # buffer for layer0 input
        self.in_dim = in_ds[weight_ids[0]]
        self.input = np.zeros((K, self.in_dim), dtype=np.float32)

        self.verbose = verbose

        #self.qconvs = []
        #self.activation_caches = []

        self.layers = []
        dilation = 1
        for weight_id in weight_ids:

            is_last_layer = (weight_id == weight_ids[-1])

            if weight_id.startswith('qconv_qb_'):
                self.layers.append(FxpMathConv1DQuantisedBitsBlock(
                    self.fxp,
                    layer_name=weight_id,
                    weights=self.weights[weight_id]['weights'][0],
                    biases=self.weights[weight_id]['weights'][1],
                    apply_relu=(not is_last_layer),
                    verbose=self.verbose
                    ))
                if not is_last_layer:
                    self.layers.append(ActivationCache(
                        depth=out_ds[weight_id],
                        dilation=K**dilation,
                        kernel_size=K
                    ))
                    dilation += 1

            elif weight_id.startswith('qconv_po2_'):
                assert (weight_id.endswith('1a') or weight_id.endswith('1b') or
                        weight_id.endswith('2a') or weight_id.endswith('2b')), weight_id

                # the last layer can only be a qconv_qb_ back down to outputs
                assert not is_last_layer

                self.layers.append(FxpMathConv1DPO2Block(
                    self.fxp,
                    layer_name=weight_id,
                    weights=self.weights[weight_id]['weights'][0],
                    biases=self.weights[weight_id]['weights'][1],
                    apply_relu=weight_id.endswith('b'),
                    verbose=self.verbose
                ))

                if weight_id.endswith('2b'):
                    self.layers.append(ActivationCache(
                        depth=out_ds[weight_id],
                        dilation=K**dilation,
                        kernel_size=K
                    ))
                    dilation += 1


    def under_and_overflow_counts(self):
        return {
            'num_underflows': sum([q.num_underflows() for q in self.layers]),
            'num_overflows': sum([q.num_overflows() for q in self.layers])
        }

    def predict(self, x):

        # convert to near fixed point numbers and back to floats
        x = self.fxp.nparray_to_fixed_point_floats(x)
        if self.verbose:
            print("==============")
            print("next_x", list(x))

        # shift input values left, and add new entry to idx -1
        for i in range(K-1):
            self.input[i] = self.input[i+1]
        self.input[K-1] = x
        if self.verbose: print("lsb", self.input)

        y_pred = self.input

        for layer in self.layers:
            if self.verbose: print("layer", layer)
            y_pred = layer.apply(y_pred)

        if self.verbose: print("y_pred", list(y_pred))
        return y_pred

    def export_weights_for_verilog(self, root_dir):
        for layer in self.layers:
            layer.export_weights_for_verilog(root_dir)
