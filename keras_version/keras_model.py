import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Add, Activation
from tensorflow.keras.models import Model
from typing import List
from pathlib import Path
import json


def create_dilated_model_from_config_and_latest_ckpt(run: str):
    run_dir_path = Path("runs") / run
    with open(run_dir_path / "model_config.json", "r") as f:
        model_config = json.load(f)
    print("model_config", model_config)
    model = create_dilated_model(**model_config)
    ckpts = (run_dir_path / "weights" / "keras").iterdir()
    latest_ckpt = list(sorted(ckpts))[-1]
    print("using ckpt", latest_ckpt)
    model.load_weights(str(latest_ckpt))
    return model


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
