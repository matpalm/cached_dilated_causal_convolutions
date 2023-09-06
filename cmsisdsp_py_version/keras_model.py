import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D
from tensorflow.keras.models import Model
from typing import List

def create_dilated_model(seq_len: int,
                         in_d: int,
                         filter_sizes: List[int],
                         kernel_size: int,
                         out_d: int,
                         all_outputs: bool=False):

    # creates a keras model that can trained to generate weights
    # for a CachedBlockModel

    inp = Input((seq_len, in_d))
    last_layer = inp

    collected_outputs = []
    for i, filter_size in enumerate(filter_sizes):

        conv_a_output = Conv1D(name=f"c{i}a", filters=filter_size,
                               kernel_size=kernel_size, dilation_rate=kernel_size**i,
                               padding='causal', activation='relu')(last_layer)
        conv_b_output = Conv1D(name=f"c{i}b", filters=filter_size,
                               kernel_size=1, strides=1,
                               activation='relu')(conv_a_output)

        collected_outputs.append(conv_b_output)
        last_layer = conv_b_output

    y_pred = Conv1D(name='y_pred', filters=out_d,
                    kernel_size=1, strides=1,
                    activation=None)(last_layer)

    if all_outputs:
        collected_outputs.append(y_pred)
        model = Model(inp, collected_outputs)
    else:
        model = Model(inp, y_pred)

    print("dilated", model.summary())
    return model


# TODO: suspect this doesn't work with an even kernel size?
#    i.e. for K=3 you can interchange dilated and strided but maybe
#    not for K=4 (?)
def create_strided_model(seq_len: int,
                         in_d: int,
                         filter_sizes: List[int],
                         kernel_size: int,
                         out_d: int):

    # creates a keras model is strided rather than dilated;
    # i.e. just runs last step inference

    inp = Input((seq_len, in_d))
    last_layer = inp

    for i, filter_size in enumerate(filter_sizes):

        conv_a_output = Conv1D(name=f"c{i}a", filters=filter_size,
                               kernel_size=kernel_size, strides=kernel_size**i,
                               padding='causal', activation='relu')(last_layer)
        conv_b_output = Conv1D(name=f"c{i}b", filters=filter_size,
                               kernel_size=1, strides=1,
                               activation='relu')(conv_a_output)
        last_layer = conv_b_output

    y_pred = Conv1D(name='y_pred', filters=out_d,
                    kernel_size=1, strides=1,
                    activation=None)(last_layer)

    model = Model(inp, y_pred)

    print("strided", model.summary())
    return model