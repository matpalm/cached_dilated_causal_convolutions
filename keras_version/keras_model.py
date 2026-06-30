import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Add, Activation
from tensorflow.keras.models import Model
from typing import List

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


def create_dilated_model(
    in_d: int,
    filter_sizes: List[int],
    kernel_size: int,
    skip_dim: int,
    out_d: int,
    all_outputs: bool = False,
):
    assert not all_outputs, "all_outputs is deprecated"

    # creates a keras model that can trained to generate weights
    # for a CachedBlockModel

    inp = Input((None, in_d))
    last_layer = inp

    skip_connections = []
    for i, filter_size in enumerate(filter_sizes):

        conv_a_output = Conv1D(name=f"c{i}a", filters=filter_size,
                               kernel_size=kernel_size, dilation_rate=kernel_size**i,
                               padding='causal', activation='relu')(last_layer)
        conv_b_output = Conv1D(name=f"c{i}b", filters=filter_size,
                               kernel_size=1, strides=1,
                               activation='relu')(conv_a_output)

        # residual connection; project input to match channels when they differ
        residual = last_layer
        if last_layer.shape[-1] != filter_size:
            residual = Conv1D(
                name=f"c{i}res",
                filters=filter_size,
                kernel_size=1,
                strides=1,
                activation=None,
            )(last_layer)

        conv_b_output = Add(name=f"c{i}add")([conv_b_output, residual])

        skip = Conv1D(
            name=f"c{i}skip",
            filters=skip_dim,
            kernel_size=1,
            strides=1,
            activation=None,
        )(conv_b_output)
        skip_connections.append(skip)

        # collected_outputs.append(conv_b_output)
        last_layer = conv_b_output

    added_skips = Add()(skip_connections)
    added_skips = Activation("relu")(added_skips)
    y_pred = Conv1D(
        name="y_pred", filters=out_d, kernel_size=1, strides=1, activation=None
    )(added_skips)

    return Model(inp, y_pred)


# TODO: suspect this doesn't work with an even kernel size?
#    i.e. for K=3 you can interchange dilated and strided but maybe
#    not for K=4 (?)
def create_strided_model(seq_len: int,
                         in_d: int,
                         filter_sizes: List[int],
                         kernel_size: int,
                         out_d: int,
                         all_outputs: bool=False):

    # creates a keras model is strided rather than dilated;
    # i.e. just runs last step inference

    inp = Input((seq_len, in_d))
    last_layer = inp

    collected_outputs = []
    for i, filter_size in enumerate(filter_sizes):

        conv_a_output = Conv1D(name=f"c{i}a", filters=filter_size,
                               kernel_size=kernel_size, strides=kernel_size**i,
                               padding='causal', activation='relu')(last_layer)
        conv_b_output = Conv1D(name=f"c{i}b", filters=filter_size,
                               kernel_size=1, strides=1,
                               activation='relu')(conv_a_output)
        collected_outputs.append(conv_b_output)
        last_layer = conv_b_output

    y_pred = Conv1D(name='y_pred', filters=out_d,
                    kernel_size=1, strides=1,
                    activation=None)(last_layer)
    collected_outputs.append(y_pred)

    if all_outputs:
        model = Model(inp, collected_outputs)
    else:
        model = Model(inp, y_pred)

    model = Model(inp, y_pred)

    print("strided", model.summary())
    return model
