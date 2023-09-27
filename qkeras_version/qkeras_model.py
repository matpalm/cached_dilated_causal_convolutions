import tensorflow as tf

import qkeras

from tensorflow.keras.layers import Input
from qkeras import quantized_bits, QConv1D, QActivation
from tensorflow.keras.models import Model
from typing import List

N_WORD = 16
N_INT = 4
N_FRAC = 12
assert N_WORD == N_INT + N_FRAC

# qkeras quantiser for all tests weights, kernels and biases
def quantiser():
    return quantized_bits(bits=N_WORD, integer=N_INT, alpha=1)

# qkeras quantiser for activation
def quant_relu():
    return f"quantized_relu({N_WORD},{N_INT})"


def masked_mse(receptive_field_size):
    def loss_fn(y_true, y_pred):
        assert len(y_true.shape) == 3, "expected (batch, sequence_length, output_dim)"
        assert y_true.shape == y_pred.shape
        # average over elements of y
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        # we want to ignore the first elements of the loss since they
        # have been fed with left padded data
        mse = mse[:,receptive_field_size:]
        # return average over batch and sequence
        return tf.reduce_mean(mse)
    return loss_fn

def create_dilated_model(seq_len: int,
                         in_out_d: int,
                         num_layers: int,
                         filter_size: int,
                         all_outputs: bool=False):

    if in_out_d != filter_size:
        raise Exception("WIP constraints; in_out_d must == filter_size.")

    # TODO: generalise code to support convs for each of....
    #   in_out_d -> filter_size     ( for first layer )
    #   filter_size -> filter_size  ( for internal layers )
    #   filter_size -> in_out_d     ( for last layer )

    # creates a qkeras model

    inp = Input((seq_len, in_out_d))
    last_layer = inp

    # TODO: first conv should be in_out_d -> filter_size
    #  but for now assuming they are the same size.

    collected_outputs = []
    for i in range(num_layers):
        conv_out = QConv1D(name=f"qconv_{i}", filters=filter_size,
                           kernel_size=4, padding='causal',
                           dilation_rate=4**i,  # fixed K=4
                           kernel_quantizer=quantiser(),
                           bias_quantizer=quantiser())(last_layer)
        relu_out = QActivation(quant_relu(), name=f"qrelu_{i}")(conv_out)

        collected_outputs.append(relu_out)
        last_layer = relu_out

    # TODO: y_pred qconv1d with filter_size -> in_out_d
    # y_pred = Conv1D(name='y_pred', filters=filter_size,
    #                 kernel_size=1, strides=1,
    #                 activation=None)(last_layer)
    # collected_outputs.append(y_pred)
    y_pred = last_layer

    if all_outputs:
        model = Model(inp, collected_outputs)
    else:
        model = Model(inp, y_pred)

    return model