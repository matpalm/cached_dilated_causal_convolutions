import unittest

import numpy as np
import tensorflow as tf
from keras.layers import *
from tensorflow.keras.optimizers import Adam

import qkeras
from qkeras import *

from fxpmath import Fxp

# a number of tests that train a qkeras model and then replicate the basic
# inference using only fxpmath operations. we do this as a fpga prototype
# because we will be able to implement the fxpmath versions exactly in
# verilog

np.set_printoptions(precision=32)

N_WORD = 16
N_INT = 4
N_FRAC = 12
assert N_WORD == N_INT + N_FRAC

# convert a value to the target fixed point representation for
# values or weights
def single_width_fxp(v):
  return Fxp(v, signed=True, n_word=N_WORD, n_frac=N_FRAC)

# convert a value to the double width target fixed point
# representation that will be used for products and accumulators
def double_width_fxp(v):
  return Fxp(v, signed=True, n_word=N_WORD*2, n_frac=N_FRAC*2)

def bits(fxp):
    return fxp.bin(frac_dot=True)

# util to convert numpy array X to float values in QI.F
def array_to_fixed_point(x):
  return float(single_width_fxp(x))
array_to_fixed_point = np.vectorize(array_to_fixed_point)

# check all values in array are QI.F
def check_all_qIF(a):
  for v in a.flatten():
    q_val = float(single_width_fxp(v))
    if v != q_val:
      raise Exception(f"value {v} not representable in QI.F; it converted to {q_val}")

# debug util to convert a numpy array X to their QI.F binary string form
def array_to_fixed_point_bin_str(x):
  return bits(single_width_fxp(x))
array_to_fixed_point_bin_str = np.vectorize(array_to_fixed_point_bin_str)




# qkeras quantiser for all tests weights, kernels and biases
def quantiser():
    return quantized_bits(bits=N_WORD, integer=N_INT, alpha=1)

# qkeras quantiser for activation
def quant_relu():
  return f"quantized_relu({N_WORD},{N_INT})"

class TestQKerasFxpMathEquivalance(unittest.TestCase):

    def _test_qkeras_dense_quantised_params_can_be_expressed_in_fxpmath(self):
        # create and train a qdense model with more data than it can handle
        # ( to force some non trivial weights ) and then check that the quantised
        # weights from the qkeras model are expressable in the fxpmath config we target

        NUM_INSTANCES = 200
        IN_D = 40
        OUT_D = 30

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



    def _test_overfit_qkeras_dense_and_custom_fxpmath_inference(self):
        # train a simple (A,B,C) input -> ((A+B)/2, -(A+C)/2)single_width_fxp output
        # that should be able to be represented by very simple quantised weights

        IN_D = 3
        OUT_D = 2
        inp = Input((IN_D,))
        single_dense = QDense(name='single_dense',
                            units=OUT_D,
                            kernel_quantizer=quantiser(),
                            bias_quantizer=quantiser(),
                            activation=None)(inp)
        single_layer_model = Model(inp, single_dense)
        single_layer_model.compile(Adam(1e-2), loss='mse')

        # generate test_x values uniformly between (-1.9, 1.9)
        # slightly less than calculated expected range (-32K, 32K) with n_int=2
        N = 100
        np.random.seed(123)
        test_x = (np.random.uniform(size=(N, IN_D))*2)-1  #  (-1, 1)
        test_x *= 1.9
        # generate test_y as ((A+B)/2), -(A+C)/2) given x as (A, B, C)
        test_y = np.empty((N, OUT_D), dtype=float)
        test_y[:,0] = (test_x[:,0] + test_x[:,1])/2
        test_y[:,1] = -(test_x[:,0] + test_x[:,2])/2
        # convert them to fixed point
        test_x = array_to_fixed_point(test_x)
        test_y = array_to_fixed_point(test_y)

        # train model. given a reasonable number of examples this model should be near perfect
        h = single_layer_model.fit(test_x, test_y, epochs=300, verbose=False)
        single_layer_model_eval_score = single_layer_model.evaluate(test_x, test_y)
        print("qkeras mse", single_layer_model_eval_score)
        self.assertTrue(single_layer_model_eval_score < 1e-5)

        # extract qkeras quantised weights
        quantised_weights = qkeras.utils.model_save_quantized_weights(single_layer_model)
        layer0_q_weights = quantised_weights['single_dense']['weights'][0]
        layer0_q_biases = quantised_weights['single_dense']['weights'][1]
        # check they are all representable fixed point
        check_all_qIF(layer0_q_weights)
        check_all_qIF(layer0_q_biases)

        # extract weights
        col0_weights = layer0_q_weights[:,0]
        col0_bias = layer0_q_biases[0]
        col1_weights = layer0_q_weights[:,1]
        col1_bias = layer0_q_biases[1]

        # create simple dot product with bias method
        def dot_product_with_bias(x, weights, bias):
            assert len(x) == len(weights)
            # init accumulator with bias value, in double width ready to be accumulated
            # with dot product results
            accum = double_width_fxp(bias)
            # this loop represents what will be in the state machine
            for i in range(len(x)):
                x_i = single_width_fxp(x[i])
                w_i = single_width_fxp(weights[i])
                prod = x_i * w_i  # recall; product will be double width
                accum += prod
            # return result resized down to single width
            accum.resize(signed=True, n_word=N_WORD, n_frac=N_FRAC)
            return accum

        # predict for single example as explicit dot product for each of
        # the two columns represent two outputs
        def predict_single(x):
            y_pred = np.array([
                # these two represent what can run in parallel
                float(dot_product_with_bias(x, col0_weights, col0_bias)),
                float(dot_product_with_bias(x, col1_weights, col1_bias))
            ])
            return y_pred

        # predicting of entire test set
        def predict_set(xs):
            return np.stack([predict_single(x) for x in xs])

        # this, like the qkeras model above, should be near perfect
        test_y_pred = predict_set(test_x)
        custom_mse = np.mean((test_y_pred-test_y) ** 2)
        print("custom_mse", custom_mse)
        self.assertTrue(custom_mse < 1e-5)



    def _test_underfit_qkeras_dense_and_custom_fxpmath_inference(self):

        # same model as above but with random data. the model will not
        # be able to fit this exactly so we expect weights to be generated
        # that are not all zeros on most significant 1/N bits
        # recall; call qkeras model(x) uses the FULL floats for the calculations
        # not the quantised weights

        IN_D = 3
        OUT_D = 2
        inp = Input((IN_D,))
        single_dense = QDense(name='single_dense',
                            units=OUT_D,
                            kernel_quantizer=quantiser(),
                            bias_quantizer=quantiser(),
                            activation=None)(inp)
        single_layer_model = Model(inp, single_dense)
        single_layer_model.compile(Adam(1e-2), loss='mse')

        # generate test_x & test_y values uniformly between (-1.9, 1.9)
        # slightly less than calculated expected range (-32K, 32K) with n_int=2
        # note: model does NOT have capacity to get this perfect
        # includes conversion to values we know are representable in QI.F
        N = 100
        np.random.seed(123)
        test_x = (np.random.uniform(size=(N, IN_D))*2)-1  #  (-1, 1)
        test_x *= 1.9
        test_x = array_to_fixed_point(test_x)
        test_y = (np.random.uniform(size=(N, OUT_D))*2)-1  #  (-1, 1)
        test_y *= 1.9
        test_y = array_to_fixed_point(test_y)

        # train model. because of random data and small model we expect this to be
        # no where near perfect
        h = single_layer_model.fit(test_x, test_y, epochs=300, verbose=False)
        single_layer_model_eval_score = single_layer_model.evaluate(test_x, test_y)
        print("qkeras mse", single_layer_model_eval_score)

        # extract qkeras quantised weights
        quantised_weights = qkeras.utils.model_save_quantized_weights(single_layer_model)
        layer0_q_weights = quantised_weights['single_dense']['weights'][0]
        layer0_q_biases = quantised_weights['single_dense']['weights'][1]
        # check they are all representable by Q2.16
        check_all_qIF(layer0_q_weights)
        check_all_qIF(layer0_q_biases)

        # extract weights
        col0_weights = layer0_q_weights[:,0]
        col0_bias = layer0_q_biases[0]
        #print("col0_weights/biases", list(map(float, col0_weights)), float(col0_bias))
        col1_weights = layer0_q_weights[:,1]
        col1_bias = layer0_q_biases[1]
        #print("col1_weights/biases", list(map(float, col1_weights)), float(col1_bias))

        # create simple dot product with bias method
        def dot_product_with_bias(x, weights, bias):
            assert len(x) == len(weights)
            # init accumulator with bias value, in double width ready to be accumulated
            # with dot product results
            accum = double_width_fxp(bias)
            #print("accum inited with bias [", bias,"] as ", bits(accum))
            # this loop represents what will be in the state machine
            for i in range(len(x)):
                x_i = single_width_fxp(x[i])
                w_i = single_width_fxp(weights[i])
                product = x_i * w_i  # recall; product will be double width
                accum += product
                # keep accumulator single width. by dft a+b => +1 for int part
                accum.resize(signed=True, n_word=2*N_WORD, n_frac=2*N_FRAC)
                #print("i", i, "x_i", bits(x_i), "w_i", bits(w_i), "=> product", bits(product), "=> accum", bits(accum), accum)

            # return result resized down to single width
            #print("accum (pre resize)  ", bits(accum), accum)
            accum.resize(signed=True, n_word=N_WORD, n_frac=N_FRAC)
            #print("accum (post resize) ", bits(accum), accum)
            return accum

        # predict for single example as explicit dot product for each of
        # the two columns represent two outputs
        def predict_single(x):
            y_pred = np.array([
                # these two represent what can run in parallel
                float(dot_product_with_bias(x, col0_weights, col0_bias)),
                float(dot_product_with_bias(x, col1_weights, col1_bias))
            ])
            return y_pred

        # predicting of entire test set
        def predict_set(xs):
            return np.stack([predict_single(x) for x in xs])

        # this, like the qkeras model above, should be near perfect
        test_y_pred = predict_set(test_x)
        custom_mse = np.mean((test_y_pred-test_y) ** 2)
        print("custom_mse", custom_mse)

        mse_diff = abs(custom_mse-single_layer_model_eval_score)
        print("mse diff", mse_diff)
        self.assertTrue(mse_diff < 1e-3)


    def test_two_layer_dense_model(self):

        # test a two layer qdense model that includes a relu activation
        # between the two

        # build model
        IN_D = 3
        HIDDEN_D = 8
        OUT_D = 2
        inp = Input((IN_D,))
        dense_0 = QDense(name='dense_0',
                            units=HIDDEN_D,
                            kernel_quantizer=quantiser(),
                            bias_quantizer=quantiser())(inp)
        dense_0 = QActivation(quant_relu(), name="dense_0_relu")(dense_0)
        dense_1 = QDense(name='dense_1',
                            units=OUT_D,
                            kernel_quantizer=quantiser(),
                            bias_quantizer=quantiser())(dense_0)
        double_layer_model = Model(inp, dense_1)
        double_layer_model.compile(Adam(1e-2), loss='mse')


        # generate test_x and test_y values uniformly between (-1.9, 1.9)
        # slightly less than calculated expected range (-32K,32K) with n_int=2
        N = 100
        test_x = (np.random.uniform(size=(N, IN_D))*2)-1  #  (-1, 1)
        test_x *= 1.9
        test_x = array_to_fixed_point(test_x)
        test_y = (np.random.uniform(size=(N, OUT_D))*2)-1  #  (-1, 1)
        test_y *= 1.9
        test_y = array_to_fixed_point(test_y)

        # train model
        h = double_layer_model.fit(test_x, test_y, epochs=200, verbose=False)
        double_layer_model_eval_score = double_layer_model.evaluate(test_x, test_y)
        print("double_layer_model_eval_score", double_layer_model_eval_score)

        # extract qkeras quantised weights
        quantised_weights = qkeras.utils.model_save_quantized_weights(double_layer_model)

        # create simple dot product with bias method
        def dot_product_with_bias(x, weights, bias, relu):
            assert len(x) == len(weights)
            # init accumulator with bias value, in double width ready to be accumulated
            # with dot product results
            accum = double_width_fxp(bias)
            # this loop represents what will be in the state machine
            for i in range(len(x)):
                x_i = single_width_fxp(x[i])
                w_i = single_width_fxp(weights[i])
                product = x_i * w_i  # recall; product will be double width
                accum += product
                # keep accumulator single width. by dft a+b => +1 for int part
                accum.resize(signed=True, n_word=2*N_WORD, n_frac=2*N_FRAC)
            # 'apply' relu, if configured
            if relu and accum < 0:
                return double_width_fxp(0)
            # return result resized down to single width
            accum.resize(signed=True, n_word=N_WORD, n_frac=N_FRAC)
            return accum

        def mat_mul_with_biases(x, weights, biases, relu):
            check_all_qIF(weights)
            check_all_qIF(biases)
            assert len(weights.shape) == 2
            assert len(biases.shape) == 1
            num_cols = weights.shape[1]
            assert weights.shape[0] == len(x)
            assert num_cols == len(biases)
            y_pred = []
            for c in range(num_cols):
                # these would all run in parallel
                y_pred.append(dot_product_with_bias(x, weights[:,c], biases[c], relu=relu))
            return np.array(y_pred)

        def predict_single(x):
            weights, biases = quantised_weights['dense_0']['weights']
            layer_0_output = mat_mul_with_biases(x, weights, biases, relu=True)
            weights, biases = quantised_weights['dense_1']['weights']
            layer_1_output = mat_mul_with_biases(layer_0_output, weights, biases, relu=False)
            return np.array(layer_1_output)

        def predict_set(xs):
            return np.stack([predict_single(x) for x in xs])

        test_y_pred = predict_set(test_x)
        custom_mse = np.mean((test_y_pred - test_y) ** 2)
        print("custom_mse", custom_mse)

        mse_diff = abs(custom_mse-double_layer_model_eval_score)
        print("mse diff", mse_diff)
        self.assertTrue(mse_diff < 1e-3)

if __name__ == '__main__':
    unittest.main(verbosity=2)