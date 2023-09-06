from typing import List
import numpy as np
import cmsisdsp as dsp

from .block import Block
from .rolling_cache import RollingCache

class Classifier(object):

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
        y_pred = self.classifier.apply(final_block_out)
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
      print('#include "classifier.h"', file=f)
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
        print(f"float layer{n}_cache_buffer[{lc.depth}*{lc.dilation}*{lc.kernel_size}];", file=f)
        print(f"RollingCache layer{n}_cache(", file=f)
        print(f"  {lc.depth}, // depth", file=f)
        print(f"  {lc.dilation}, // dilation", file=f)
        print(f"  {lc.kernel_size}, // kernel size", file=f)
        print(f"  layer{n}_cache_buffer", file=f)
        print(f");", file=f)
        print("", file=f)

      print(f"float classifier_weights{ca(self.classifier.weights)}", file=f)
      print(f"float classifier_biases{ca(self.classifier.biases)}", file=f)
      print(f"Classifier classifier(", file=f)
      print(f"  {self.classifier.input_dim}, // input_dim", file=f)
      print(f"  {self.classifier.output_dim}, // output_dim", file=f)
      print(f"  classifier_weights,", file=f)
      print(f"  classifier_biases", file=f)
      print(f");", file=f)
