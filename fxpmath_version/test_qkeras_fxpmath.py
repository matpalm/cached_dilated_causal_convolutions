import unittest

import numpy as np
import tensorflow as tf
from keras.layers import *
from tensorflow.keras.optimizers import Adam

import qkeras
from qkeras import *

from fxpmath import Fxp

# Some utils

N_WORD = 16
N_INT = 4
N_FRAC = 12
assert N_WORD == N_INT + N_FRAC

def single_width_fxp(v):
  return Fxp(v, signed=True, n_word=N_WORD, n_frac=N_FRAC)

def double_width_fxp(v):
  return Fxp(v, signed=True, n_word=N_WORD*2, n_frac=N_FRAC*2)

# util to convert numpy array X to float values in QI.F
def to_fixed_point(x):
  return float(Fxp(x, signed=True, n_word=N_WORD, n_frac=N_FRAC))
to_fixed_point = np.vectorize(to_fixed_point)

# check all values in array are QI.F
def check_all_qIF(a):
  for v in a.flatten():
    q_val = float(Fxp(v, signed=True, n_word=N_WORD, n_frac=N_FRAC))
    if v != q_val:
      raise Exception(f"value {v} not representable in QI.F; it converted to {q_val}")

# debug util to convert a numpy array X to their QI.F binary string form
def to_fixed_point_bin_str(x):
  return Fxp(x, signed=True, n_word=N_WORD, n_frac=N_FRAC).bin(frac_dot=True)
to_fixed_point_bin_str = np.vectorize(to_fixed_point_bin_str)


class TestQKerasFxpMathEquivalance(unittest.TestCase):

    def test_qkeras_dense_quantised_params_can_be_expressed_in_fxpmath(self):
        # create and train a qdense model with more data than it can handle
        # ( to force some non trivial weights ) and then check that the quantised
        # weights from the model are expressable in the fxpmath config we target

        NUM_INSTANCES = 200
        IN_D = 40
        OUT_D = 30

        def quantiser():
            return quantized_bits(bits=N_WORD, integer=N_INT, alpha=1)

        # create a simple qkeras model of a single qdense layer
        inp = Input((IN_D,))
        single_dense = QDense(name='single_dense',
                            units=OUT_D,
                            kernel_quantizer=quantiser(),
                            bias_quantizer=quantiser(),
                            activation=None)(inp)
        single_layer_model = Model(inp, single_dense)
        single_layer_model.compile(Adam(1e-2), loss='mse')

        # run some random data through it. note: we don't expect this to overfit to
        # be perfect.
        np.random.seed(123)
        test_x = (np.random.uniform(size=(NUM_INSTANCES, IN_D))*2)-1  #  (-1, 1)
        test_y = (np.random.uniform(size=(NUM_INSTANCES, OUT_D))*2)-1  #  (-1, 1)
        _ = single_layer_model.fit(test_x, test_y, epochs=100, verbose=False)

        # extract quantised weights from qkeras model
        quantised_weights = qkeras.utils.model_save_quantized_weights(single_layer_model)
        layer0_q_weights = quantised_weights['single_dense']['weights'][0]
        layer0_q_biases = quantised_weights['single_dense']['weights'][1]

        # check they are all representable with fxp math
        check_all_qIF(layer0_q_weights)
        check_all_qIF(layer0_q_biases)



if __name__ == '__main__':
    unittest.main()