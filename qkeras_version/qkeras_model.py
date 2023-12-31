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
        return quantized_po2(bits=N_WORD, max_value=1)
    else:
        return quantized_bits(bits=N_WORD, integer=N_INT, alpha=1)

# qkeras quantiser for all convolution activations
def quant_relu(po2=False):
    if po2:
        assert False, "never need to use this?"
        return f"quantized_po2({N_WORD}, 1)"
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


class QKerasModelBuilder(object):

    def __init__(self):
        self.layer_info = []

    def add_quantized_bits_conv_block(
            self,
            inp,
            layer_number: int,     # for dilation amount & naming
            out_filters: int,
            l2: float,
            relu: bool
        ):

        layer_id = f"qconv_{layer_number}_qb"
        y_pred = QConv1D(name=layer_id,
                        filters=out_filters,
                        kernel_size=K, padding='causal',
                        dilation_rate=K**layer_number,
                        kernel_quantizer=quantiser(),
                        bias_quantizer=quantiser(),
                        kernel_regularizer=regularizers.L2(l2),
                        bias_regularizer=regularizers.L2(l2))(inp)
        self.layer_info.append({'type': 'qb', 'id': layer_id})

        if relu:
            y_pred = QActivation(quant_relu(), name=f"qrelu_{layer_number}")(y_pred)
            self.layer_info.append({'type': 'relu'})

        return y_pred

    def add_quantized_po2_conv_block(
            self,
            inp,
            layer_number: int,     # for dilation amount & naming
            l2: float,
            out_filters: int,
            po2_filters: int
        ):
            # start with a _qb conv layer to handle the dilation
            layer_id = f"qconv_{layer_number}_qb"
            y_pred = QConv1D(name=layer_id,
                            filters=out_filters,
                            kernel_size=K, padding='causal',
                            dilation_rate=K**layer_number,
                            kernel_quantizer=quantiser(),
                            bias_quantizer=quantiser(),
                            kernel_regularizer=regularizers.L2(l2),
                            bias_regularizer=regularizers.L2(l2))(inp)
            self.layer_info.append({'type': 'qb', 'id': layer_id})

            y_pred = QActivation(quant_relu(), name=f"qrelu_{layer_number}")(y_pred)
            self.layer_info.append({'type': 'relu'})

            # then a pair of 1x1 _po2 convs; expand to po2_filters, contract back to out_filters
            for sublayer in [1, 2]:

                layer_id = f"qconv_{layer_number}_{sublayer}a_po2"
                y_pred = QConv1D(name=layer_id,
                                filters=po2_filters,
                                kernel_size=1, padding='valid',
                                kernel_quantizer=quantiser(po2=True),
                                bias_quantizer=quantiser())(y_pred)
                self.layer_info.append({'type': 'po2', 'id': layer_id})

                layer_id = f"qconv_{layer_number}_{sublayer}b_po2"
                y_pred = QConv1D(name=layer_id,
                                filters=out_filters,
                                kernel_size=1, padding='valid',
                                kernel_quantizer=quantiser(po2=True),
                                bias_quantizer=quantiser())(y_pred)
                self.layer_info.append({'type': 'po2', 'id': layer_id})

                y_pred = QActivation(quant_relu(), name=f"qrelu_{layer_number}_{sublayer}")(y_pred)
                self.layer_info.append({'type': 'relu'})

            return y_pred


    def create_dilated_model(
            self,
            seq_len: int,
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

        self.layer_info = []

        inp = Input((seq_len, in_out_d))

        y_pred = self.add_quantized_bits_conv_block(inp, layer_number=0,
            out_filters=filter_size, l2=l2, relu=True)

        self.layer_info.append({'type': 'dilation', 'amount': K, 'depth': filter_size})

        if po2_filter_size is None:
            # LKG "standard" model
            y_pred = self.add_quantized_bits_conv_block(y_pred, layer_number=1,
                out_filters=filter_size, l2=l2, relu=True)
        else:
            # using po2
            y_pred = self.add_quantized_po2_conv_block(y_pred, layer_number=1,
                out_filters=filter_size, po2_filters=po2_filter_size, l2=l2)

        self.layer_info.append({'type': 'dilation', 'amount': K*K, 'depth': filter_size})

        y_pred = self.add_quantized_bits_conv_block(y_pred, layer_number=2,
            out_filters=in_out_d, l2=l2, relu=False)

        return Model(inp, y_pred)
