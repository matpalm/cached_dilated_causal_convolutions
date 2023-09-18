from typing import List
import numpy as np
import cmsisdsp as dsp

from .block import Block
from .rolling_cache import RollingCache

class Regression(object):

    def __init__(self, weights, biases):
        assert len(weights.shape) == 2
        self.input_dim = weights.shape[0]
        self.output_dim = weights.shape[1]
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
                 regression: Regression):

        self.blocks = blocks
        self.regression = regression

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

        # shift input values left, and add new entry to idx -1
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
        y_pred = self.regression.apply(final_block_out)
        return y_pred

    def write_model_defn_h(self, f):
        # may god have mercy on my soul for this method :/

        def ca(a):
            shapes_as_product = "*".join(map(str, a.shape))
            return "[" + shapes_as_product + "] = {" + ", ".join(map(str, a.flatten().tolist())) + "};"

        print('#pragma once', file=f)
        print('#include "left_shift_buffer.h"', file=f)
        print('#include "block.h"', file=f)
        print('#include "rolling_cache.h"', file=f)
        print('#include "regression.h"', file=f)
        print("", file=f)

        print("LeftShiftBuffer left_shift_input_buffer(", file=f)
        print(f"    {self.kernel_size},   // kernel size", file=f)
        print(f"    {self.input_feature_depth});  // feature depth", file=f)
        print("", file=f)

        for n, block in enumerate(self.blocks):
            print(f"float b{n}_c1_kernel{ca(block.c1_kernel)}", file=f)
            print(f"float b{n}_c1_bias{ca(block.c1_bias)}", file=f)
            print(f"float b{n}_c2_kernel{ca(block.c2_kernel)}", file=f)
            print(f"float b{n}_c2_bias{ca(block.c2_bias)}", file=f)
            print(f"Block block{n}({block.kernel_size}, // kernel_size", file=f)
            print(f"             {block.in_d}, {block.c2_out_d}, // in_d, out_d", file=f)
            print(f"             b{n}_c1_kernel, b{n}_c1_bias, b{n}_c2_kernel, b{n}_c2_bias);", file=f)
            print("", file=f)

        for n, lc in enumerate(self.layer_caches):
            print(f"const size_t layer{n}_depth = {lc.depth};", file=f)
            print(f"const size_t layer{n}_dilation = {lc.dilation};", file=f)
            print(f"const size_t layer{n}_kernel_size = {lc.kernel_size};", file=f)
            print(f"float layer{n}_cache_buffer[layer{n}_dilation * layer{n}_kernel_size * layer{n}_depth];", file=f)
            print(f"RollingCache layer{n}_cache(layer{n}_depth, layer{n}_dilation,"
                  f" layer{n}_kernel_size, layer{n}_cache_buffer);", file=f)
            print("", file=f)

        print(f"float regression_weights{ca(self.regression.weights)}", file=f)
        print(f"float regression_biases{ca(self.regression.biases)}", file=f)
        print(f"Regression regression(", file=f)
        print(f"  {self.regression.input_dim}, // input_dim", file=f)
        print(f"  {self.regression.output_dim}, // output_dim", file=f)
        print(f"  regression_weights,", file=f)
        print(f"  regression_biases", file=f)
        print(f");", file=f)


def create_cached_block_model_from_keras_model(keras_model, input_feature_depth):
    from tensorflow.keras.layers import InputLayer, Conv1D

    assert type(keras_model.layers[0]) == InputLayer

    blocks = []
    i = 1
    while i+1 < len(keras_model.layers):
        # next two layer should be 1) dilated conv and then 2) 1x1 conv

        block_c1 = keras_model.layers[i]
        block_c2 = keras_model.layers[i+1]
        assert type(block_c1) == Conv1D
        assert type(block_c2) == Conv1D

        # these two make up the next block
        blocks.append(Block(
            c1_kernel = block_c1.weights[0].numpy(),
            c1_bias = block_c1.weights[1].numpy(),
            c2_kernel = block_c2.weights[0].numpy(),
            c2_bias = block_c2.weights[1].numpy(),
        ))

        i += 2

    # final layer is the regression
    regression = Regression(
        weights=keras_model.layers[-1].weights[0].numpy()[0],
        biases=keras_model.layers[-1].weights[1].numpy()
    )

    # create CachedBlockModel since it creates correct layer
    # caches
    return CachedBlockModel(
        blocks=blocks,
        input_feature_depth=input_feature_depth,
        regression=regression
    )
