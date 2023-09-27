import unittest

import numpy as np
import tensorflow as tf
from keras.layers import *
from tensorflow.keras.optimizers import Adam

import qkeras
from qkeras import *

from fxpmath import Fxp

from .fxpmath_conv1d import FxpMathConv1D
from .util import FxpUtil

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


def resize_single_width(v):
    v.resize(signed=True, n_word=N_WORD, n_frac=N_FRAC)

def resize_double_width(v):
    v.resize(signed=True, n_word=N_WORD*2, n_frac=N_FRAC*2)


# qkeras quantiser for all tests weights, kernels and biases
def quantiser():
    return quantized_bits(bits=N_WORD, integer=N_INT, alpha=1)

# qkeras quantiser for activation
def quant_relu():
  return f"quantized_relu({N_WORD},{N_INT})"

def init_seeds(s):
    tf.random.set_seed(s)
    np.random.seed(s)
    import random
    random.seed(s)

def qkeras_custom_mse_equivalant(test_x, test_y, qkeras_model, custom_inference_fn, atol=1e-3):

    # evaluate qkeras_model mse
    qkeras_mse = qkeras_model.evaluate(test_x, test_y)

    # run all examples through custom function
    def predict_set(xs):
        return np.stack([custom_inference_fn(x) for x in xs])

    # this, like the qkeras model above, should be near perfect
    test_y_pred = predict_set(test_x)
    custom_mse = np.mean((test_y_pred - test_y) ** 2)
    print("custom_mse", custom_mse)

    mse_diff = abs(qkeras_mse - custom_mse)
    print("qkeras_mse ", qkeras_mse, "custom_mse", custom_mse, "=> mse diff", mse_diff)
    return mse_diff < atol



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
        _h = single_layer_model.fit(test_x, test_y, epochs=300, verbose=False)

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
            resize_single_width(accum)
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

        self.assertTrue(
            qkeras_custom_mse_equivalant(test_x, test_y, single_layer_model, predict_single))



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
        _h = single_layer_model.fit(test_x, test_y, epochs=300, verbose=False)

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
                # keep accumulator double width. by dft a+b => +1 for int part
                resize_double_width(accum)
                #print("i", i, "x_i", bits(x_i), "w_i", bits(w_i), "=> product", bits(product), "=> accum", bits(accum), accum)

            # return result resized down to single width
            #print("accum (pre resize)  ", bits(accum), accum)
            resize_single_width(accum)
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

        self.assertTrue(
            qkeras_custom_mse_equivalant(test_x, test_y, single_layer_model, predict_single))


    def _test_two_layer_dense_model(self):
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
        _h = double_layer_model.fit(test_x, test_y, epochs=200, verbose=False)

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
                resize_double_width(accum)
            # 'apply' relu, if configured
            if relu and accum < 0:
                return double_width_fxp(0)
            # return result resized down to single width
            resize_single_width(accum)
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

        self.assertTrue(
            qkeras_custom_mse_equivalant(test_x, test_y, double_layer_model, predict_single))


    def _test_qkeras_conv1d(self):
        raise Exception("replaced with test_qkeras_conv1d_object")

        N = 5
        K = 4
        IN_D = 3
        OUT_D = 2

        inp = Input((K, IN_D))
        qconv1d = QConv1D(name='conv1', filters=OUT_D,
                        kernel_size=K, strides=1,
                        padding='causal', dilation_rate=1,
                        kernel_quantizer=quantiser(),
                        bias_quantizer=quantiser())(inp)
        # take last from sequence for loss
        qconv1d = qconv1d[:, -1, :]
        single_conv_model = Model(inp, qconv1d)
        single_conv_model.compile(Adam(1e-2), loss='mse')

        # generate test_x and test_y values uniformly between (-1.9, 1.9)
        # slightly less than calculated expected range (-32K,32K) with n_int=2
        N = 100
        test_x = (np.random.uniform(size=(N, K, IN_D))*2)-1  #  (-1, 1)
        test_x *= 1.9
        test_x = array_to_fixed_point(test_x)
        test_y = (np.random.uniform(size=(N, OUT_D))*2)-1  #  (-1, 1)
        test_y *= 1.9
        test_y = array_to_fixed_point(test_y)

        # fit model
        _ = single_conv_model.fit(test_x, test_y, epochs=200, verbose=False)

        # extract qkeras quantised weights
        quantised_weights = qkeras.utils.model_save_quantized_weights(single_conv_model)
        quantised_weights
        conv1_weights = quantised_weights['conv1']['weights'][0]
        conv1_biases = quantised_weights['conv1']['weights'][1]

        def dot_product(x, weights, accumulator):
            assert len(x.shape) == 1
            assert len(weights.shape) == 1
            assert len(x) == len(weights)
            # this loop represents what will be in the state machine
            for i in range(len(x)):
                x_i = single_width_fxp(x[i])
                w_i = single_width_fxp(weights[i])
                product = x_i * w_i  # will be double width
                accumulator += product
                # keep accumulator double width. by dft a+b => +1 for int part
                resize_double_width(accumulator)
            return accumulator

        def row_by_matrix_multiply(x, weights, accumulators):
            assert len(x.shape) == 1
            in_d = x.shape[0]
            assert len(weights.shape) == 2
            assert weights.shape[0] == in_d
            out_d = weights.shape[1]
            assert len(accumulators) == out_d
            for column in range(out_d):
                accumulators[column] = dot_product(x, weights[:, column], accumulators[column])
            return accumulators

        def conv_1d_mat_mul_with_biases(x, weights, biases, relu):
            check_all_qIF(weights)
            check_all_qIF(biases)
            assert len(x.shape) == 2
            assert x.shape[0] == 4  # K
            in_d = x.shape[1]
            assert len(weights.shape) == 3
            assert weights.shape[0] == 4  # K
            assert weights.shape[1] == in_d, f"{weights.shape[1]}!={in_d}"
            out_d = weights.shape[2]
            assert len(biases.shape) == 1
            assert len(biases) == out_d

            # initialise accumulators using the biases
            # note: double width for accumulation
            accumulators = [ double_width_fxp(b) for b in biases ]

            # run, and accumulate, for each kernel/x  note: these can all run
            # in parallel, but in that case we'd need another accumulation and
            # would only need to initial the first with the biases
            for k in range(4):
                # these can all run in parallel
                accumulators = row_by_matrix_multiply(x[k], weights[k], accumulators)

            # resize down from double to single for output
            for i in range(out_d):
                resize_single_width(accumulators[i])

            # apply relu, if configured
            if relu:
                for i in range(out_d):
                    if accumulator[i] < 0:
                        accumulator[i] = 0

            # return as np array,
            return np.array(accumulators)

        def predict_single(x):
            return conv_1d_mat_mul_with_biases(x, conv1_weights, conv1_biases, relu=False)

        self.assertTrue(
            qkeras_custom_mse_equivalant(test_x, test_y, single_conv_model, predict_single))




    def test_qkeras_conv1d_object(self):

        # single qkeras dense layer but with custom inference more like what
        # we'll want to run in verilog

        init_seeds(234)
        N = 5
        K = 4
        IN_D = 3
        OUT_D = 2
        inp = Input((K, IN_D))
        qconv1d = QConv1D(name='conv1', filters=OUT_D,
                        kernel_size=K, strides=1,
                        padding='causal', dilation_rate=1,
                        kernel_quantizer=quantiser(),
                        bias_quantizer=quantiser())(inp)
        # take last from sequence for loss
        qconv1d = qconv1d[:, -1, :]
        single_conv_model = Model(inp, qconv1d)
        single_conv_model.compile(Adam(1e-2), loss='mse')

        # generate test_x and test_y values uniformly between (-1.9, 1.9)
        # slightly less than calculated expected range (-32K,32K) with n_int=2
        N = 100
        test_x = (np.random.uniform(size=(N, K, IN_D))*2)-1  #  (-1, 1)
        test_x *= 1.9
        test_x = array_to_fixed_point(test_x)
        test_y = (np.random.uniform(size=(N, OUT_D))*2)-1  #  (-1, 1)
        test_y *= 1.9
        test_y = array_to_fixed_point(test_y)

        # fit model
        _ = single_conv_model.fit(test_x, test_y, epochs=200, verbose=False)

        # extract qkeras quantised weights
        quantised_weights = qkeras.utils.model_save_quantized_weights(single_conv_model)
        conv1_weights = quantised_weights['conv1']['weights'][0]
        conv1_biases = quantised_weights['conv1']['weights'][1]

        # wrap in FxpMathConv1D object for inference using fxp math
        fxp_util = FxpUtil(n_word=N_WORD, n_int=N_INT, n_frac=N_FRAC)
        fxp_conv1d = FxpMathConv1D(fxp_util, conv1_weights, conv1_biases)
        fxp_conv1d.export_weights_per_dot_product('/tmp/test_qkeras_conv1d_object.hex')

        def predict_single(x):
            return fxp_conv1d.apply(x, relu=False)

        self.assertTrue(
            qkeras_custom_mse_equivalant(test_x, test_y, single_conv_model, predict_single))

    def _test_weight_export_from_FxpMathConv1D(self):
        # create dummy model
        init_seeds(234)
        N = 5
        K = 4
        IN_D = 2
        OUT_D = 3
        inp = Input((K, IN_D))
        qconv1d = QConv1D(name='conv1', filters=OUT_D,
                kernel_size=K, strides=1,
                padding='causal', dilation_rate=1,
                kernel_quantizer=quantiser(),
                bias_quantizer=quantiser())(inp)
        qconv1d = qconv1d[:, -1, :]
        single_conv_model = Model(inp, qconv1d)

        # convert (untrained) weights to qkeras values
        quantised_weights = qkeras.utils.model_save_quantized_weights(single_conv_model)
        conv1_weights = quantised_weights['conv1']['weights'][0]
        conv1_biases = quantised_weights['conv1']['weights'][1]
        check_all_qIF(conv1_weights)
        check_all_qIF(conv1_biases)

        # construct fxpmath inference wrapper
        fxp_util = FxpUtil(n_word=N_WORD, n_int=N_INT, n_frac=N_FRAC)
        fxp_conv1d = FxpMathConv1D(fxp_util, conv1_weights, conv1_biases)
        fxp_conv1d.export_weights_per_dot_product('/tmp/foo.hex')

        # todo; assert something about /tmp/foo.hex

if __name__ == '__main__':
    unittest.main(verbosity=2)