from typing import List
import numpy as np

from .block import Block
from .rolling_cache import RollingCache

class CachedBlockModel(object):
    def __init__(self,
                 blocks: List[Block],
                 input_feature_depth: int,
                 classifier_kernel,
                 classifier_bias):

        self.blocks = blocks
        self.classifier_kernel = classifier_kernel
        self.classifier_bias = classifier_bias

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
        # TODO: port to arm math too
        y_pred = np.dot(final_block_out, self.classifier_kernel.squeeze()) + self.classifier_bias
        return y_pred