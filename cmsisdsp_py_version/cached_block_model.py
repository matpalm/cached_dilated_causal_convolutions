from typing import List
import numpy as np
import cmsisdsp as dsp

from .block import Block
from .rolling_cache import RollingCache

class Classifier(object):

    def __init__(self, weights, biases):
        assert len(weights.shape) == 2
        self.input_dim = weights.shape[0]
        assert biases.shape == (weights.shape[1],)
        self.weights = weights
        self.biases = biases

    def apply(self, x):
        assert x.shape == (self.input_dim,)
        x_mi = x.reshape((1, self.input_dim))
        weights_mi = self.weights
        _status, result = dsp.arm_mat_mult_f32(x_mi, weights_mi)
        return dsp.arm_add_f32(result, self.biases)


class CachedBlockModel(object):

    def __init__(self,
                 blocks: List[Block],
                 input_feature_depth: int,
                 classifier: Classifier):

        self.blocks = blocks
        self.classifier = classifier

        self.kernel_size = blocks[0].kernel_size
        self.input_feature_depth = input_feature_depth

        # buffer for layer0 input
        self.input = np.zeros((self.kernel_size,
                               self.input_feature_depth), dtype=np.float32)

        # create layer cache for each block except the last ( which
        # is just run once each inference so doesn't need caching
        self.layer_caches = []
        for i in range(len(self.blocks)-1):
            self.layer_caches.append(
                RollingCache(
                    depth=self.blocks[i].output_feature_depth(),
                    dilation=self.kernel_size ** (i+1),
                    kernel_size=self.kernel_size))

    def apply(self, x):
        assert x.shape == (self.input_feature_depth,), x.shape

        # clumsy shift input, will do something better in firmware
        for i in range(self.kernel_size-1):
            self.input[i] = self.input[i+1]
        self.input[self.kernel_size-1] = x

        # for each block, except the last, run the block, add to
        # cache and then derive input for next layer
        feature_map = self.input
        for i in range(len(self.blocks)-1):
            block_output = self.blocks[i].apply(feature_map)
            self.layer_caches[i].add(block_output)
            feature_map = self.layer_caches[i].cached_dilated_values()

        # only ever need to run final dilation layer once
        final_block_out = self.blocks[-1].apply(feature_map)

        # run y_pred
        y_pred = self.classifier.apply(final_block_out)
        return y_pred