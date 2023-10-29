import tensorflow as tf

import qkeras

from tensorflow.keras.layers import Input
from qkeras import quantized_bits, quantized_po2, QConv1D, QActivation
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from typing import List

N_WORD = 16
N_INT = 4
N_FRAC = 12
assert N_WORD == N_INT + N_FRAC

# qkeras quantiser for all convolution kernels and biases
def quantiser(po2=False):
    if po2:
        return quantized_po2(bits=N_WORD, max_value=2**N_INT)
    else:
        return quantized_bits(bits=N_WORD, integer=N_INT, alpha=1)

# qkeras quantiser for all convolution activations
def quant_relu(po2=False):
    if po2:
        assert False, "never need to use this?"
        return f"quantized_po2({N_WORD}, {2**N_INT})"
    else:
        return f"quantized_relu({N_WORD},{N_INT})"


def masked_mse(receptive_field_size, filter_column_idx=None):
    '''
    Calculates masked version of mean square error

    Parameters:
        receptive_field_size: the number of leading elements to ignore in loss as
                              these are "polluted" by the 0 padding during training.
        filter_column_idx: only calculate loss w.r.t this column in output. done since
                           the output has 4 outs, but we might only care about one.
    Returns:
        keras loss function
    '''

    def loss_fn(y_true, y_pred):
        assert len(y_true.shape) == 3, "expected (batch, sequence_length, output_dim)"
        if filter_column_idx is not None:
            # consider only a single column from output for loss
            y_true = y_true[:,:,filter_column_idx:filter_column_idx+1]
            y_pred = y_pred[:,:,filter_column_idx:filter_column_idx+1]
        assert y_true.shape == y_pred.shape
        # average over elements of y
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        # we want to ignore the first elements of the loss since they
        # have been fed with left padded data
        mse = mse[:,receptive_field_size:]
        # return average over batch and sequence
        return tf.reduce_mean(mse)
    return loss_fn

K = 4

def add_quantized_bits_conv_block(
        inp,
        layer_number: int,     # for dilation amount & naming
        out_filters: int,
        l2: float,
        relu: bool
    ):

    y_pred = QConv1D(name=f"qconv_{layer_number}_qb",
                     filters=out_filters,
                     kernel_size=K, padding='causal',
                     dilation_rate=K**layer_number,
                     kernel_quantizer=quantiser(),
                     bias_quantizer=quantiser(),
                     kernel_regularizer=regularizers.L2(l2),
                     bias_regularizer=regularizers.L2(l2))(inp)
    if relu:
        y_pred = QActivation(quant_relu(), name=f"qrelu_{layer_number}")(y_pred)
    return y_pred

def add_quantized_po2_conv_block(
        inp,
        layer_number: int,     # for dilation amount & naming
        out_filters: int,
        po2_filters: int
    ):
        y_pred = QConv1D(name=f"qconv_{layer_number}_1a_po2",
                         filters=po2_filters,
                         kernel_size=K, padding='causal',
                         dilation_rate=K**layer_number,
                         kernel_quantizer=quantiser(po2=True),
                         bias_quantizer=quantiser())(inp)
        y_pred = QConv1D(name=f"qconv_{layer_number}_1b_po2",
                         filters=out_filters,
                         kernel_size=1, padding='valid',
                         kernel_quantizer=quantiser(po2=True),
                         bias_quantizer=quantiser())(y_pred)
        y_pred = QActivation(quant_relu(), name=f"qrelu_{layer_number}_1")(y_pred)
        y_pred = QConv1D(name=f"qconv_{layer_number}_2a_po2",
                         filters=po2_filters,
                         kernel_size=1, padding='valid',
                         kernel_quantizer=quantiser(po2=True),
                         bias_quantizer=quantiser())(y_pred)
        y_pred = QConv1D(name=f"qconv_{layer_number}_2b_po2",
                         filters=out_filters,
                         kernel_size=1, padding='valid',
                         kernel_quantizer=quantiser(po2=True),
                         bias_quantizer=quantiser())(y_pred)
        y_pred = QActivation(quant_relu(), name=f"qrelu_{layer_number}_2")(y_pred)
        return y_pred


def create_dilated_model(seq_len: int,
                         in_out_d: int,
                         num_layers: int,
                         filter_size: int,
                         po2_filter_size: int,
                         l2: float=0.0):
    '''
    create a qkeras model with a stack of dilation 1d convolutions

    Parameters:
        seq_len: the length of the input sequence.
        in_out_d: the feature dim of both the input and the output)
        num_layers: number of 1d convolution to stack, each with an increasing dilation
        filter_size: kernel size for each convolution
        l2: l2 penality for convolution kerne & bias
    Returns:
        qkeras model
    '''

    assert num_layers == 3, "wip refactoring re: po2 layers"

    inp = Input((seq_len, in_out_d))

    y_pred = add_quantized_bits_conv_block(inp, layer_number=0,
        out_filters=filter_size, l2=l2, relu=True)

    if po2_filter_size is None:
        # LKG "standard" model
        y_pred = add_quantized_bits_conv_block(y_pred, layer_number=1,
            out_filters=filter_size, l2=l2, relu=True)
    else:
        # using po2
        y_pred = add_quantized_po2_conv_block(y_pred, layer_number=1,
            out_filters=filter_size, po2_filters=po2_filter_size)

    y_pred = add_quantized_bits_conv_block(y_pred, layer_number=2,
        out_filters=in_out_d, l2=l2, relu=False)

    return Model(inp, y_pred)
