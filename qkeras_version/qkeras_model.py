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
def quantiser(po2):
    if po2:
        return quantized_po2(bits=N_WORD, max_value=2**N_INT)
    else:
        return quantized_bits(bits=N_WORD, integer=N_INT, alpha=1)

# qkeras quantiser for all convolution activations
def quant_relu(po2):
    if po2:
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

def create_dilated_model(seq_len: int,
                         in_out_d: int,
                         num_layers: int,
                         filter_size: int,
                         l2: float=0.0,
                         all_outputs: bool=False):
    '''
    create a qkeras model with a stack of dilation 1d convolutions

    Parameters:
        seq_len: the length of the input sequence.
        in_out_d: the feature dim of both the input and the output)
        num_layers: number of 1d convolution to stack, each with an increasing dilation
        filter_size: kernel size for each convolution
        l2: l2 penality for convolution kerne & bias
        all_outputs: if true return a model that outputs all layers for debugging.
                     otherwise just return the final output
    Returns:
        qkeras model
    '''

    inp = Input((seq_len, in_out_d))

    K = 4

    y_pred = QConv1D(name=f"qconv_0",
                        filters=16,
                        kernel_size=K, padding='causal',
                        dilation_rate=K**0,
                        kernel_quantizer=quantiser(po2=False),
                        bias_quantizer=quantiser(po2=False),
                        kernel_regularizer=regularizers.L2(l2),
                        bias_regularizer=regularizers.L2(l2))(inp)
    y_pred = QActivation(quant_relu(po2=False), name=f"qrelu_0")(y_pred)

    y_pred = QConv1D(name=f"qconv_1a",
                        filters=256,
                        kernel_size=K, padding='causal',
                        dilation_rate=K**1,
                        kernel_quantizer=quantiser(po2=True),
                        bias_quantizer=quantiser(po2=True),
                        kernel_regularizer=regularizers.L2(l2),
                        bias_regularizer=regularizers.L2(l2))(y_pred)
    y_pred = QConv1D(name=f"qconv_1b",
                        filters=256,
                        kernel_size=1,
                        kernel_quantizer=quantiser(po2=True),
                        bias_quantizer=quantiser(po2=True),
                        kernel_regularizer=regularizers.L2(l2),
                        bias_regularizer=regularizers.L2(l2))(y_pred)
    y_pred = QActivation(quant_relu(po2=True), name=f"qrelu_1a")(y_pred)
    y_pred = QConv1D(name=f"qconv_1c",
                        filters=256,
                        kernel_size=1,
                        kernel_quantizer=quantiser(po2=True),
                        bias_quantizer=quantiser(po2=True),
                        kernel_regularizer=regularizers.L2(l2),
                        bias_regularizer=regularizers.L2(l2))(y_pred)
    y_pred = QConv1D(name=f"qconv_1d",
                        filters=256,
                        kernel_size=1,
                        kernel_quantizer=quantiser(po2=True),
                        bias_quantizer=quantiser(po2=True),
                        kernel_regularizer=regularizers.L2(l2),
                        bias_regularizer=regularizers.L2(l2))(y_pred)
    y_pred = QActivation(quant_relu(po2=True), name=f"qrelu_1b")(y_pred)

    y_pred = QConv1D(name=f"qconv_2",
                        filters=4,
                        kernel_size=K, padding='causal',
                        dilation_rate=K**2,
                        kernel_quantizer=quantiser(po2=False),
                        bias_quantizer=quantiser(po2=False),
                        kernel_regularizer=regularizers.L2(l2),
                        bias_regularizer=regularizers.L2(l2))(y_pred)

    return Model(inp, y_pred)

    # collected_outputs = []
    # for i in range(num_layers):
    #     is_last_layer = i == num_layers-1
    #     last_layer = QConv1D(name=f"qconv_{i}",
    #                         filters=in_out_d if is_last_layer else filter_size,
    #                         kernel_size=K, padding='causal',
    #                         dilation_rate=K**i,
    #                         kernel_quantizer=quantiser(),
    #                         bias_quantizer=quantiser(),
    #                         kernel_regularizer=regularizers.L2(l2),
    #                         bias_regularizer=regularizers.L2(l2))(last_layer)
    #     collected_outputs.append(last_layer)

    #     if not is_last_layer:
    #         last_layer = QActivation(quant_relu(), name=f"qrelu_{i}")(last_layer)
    #         collected_outputs.append(last_layer)

    # y_pred = last_layer

    # if all_outputs:
    #     model = Model(inp, collected_outputs)
    # else:
    #     model = Model(inp, y_pred)

    return model