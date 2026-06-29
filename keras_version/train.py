import pandas as pd
import numpy as np
import random
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json

from tensorflow.keras.optimizers import Adam

from .keras_model import create_dilated_model, masked_mse
# from cmsisdsp_py_version.cached_block_model import create_cached_block_model_from_keras_model

# from tf_data_pipeline.data import Embed2DWaveFormData
from tf_data_pipeline.pcapture_data import ParametricCaptureData
from .util import CheckYPred

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-data-z", type=Path, required=True)
    parser.add_argument("--num-train-batches", type=int, default=1_000)
    parser.add_argument("--num-validate-batches", type=int, default=10)
    parser.add_argument("--cache-fname", type=str, default=None)
    opts = parser.parse_args()
    print("opts", opts)

    tensorboard_dir = "runs" / opts.run / "tb"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = "runs" / opts.run / "weights" / "keras"
    weights_dir.mkdir(parents=True, exist_ok=True)

    IN_D = 4  # triangle, 3 cvs
    OUT_D = 1  # output wave
    K = 4
    FILTER_SIZES = [8, 16, 32, 64, 128]
    RECEPTIVE_FIELD_SIZE = K ** len(FILTER_SIZES)
    TRAIN_SEQ_LEN = RECEPTIVE_FIELD_SIZE * 5
    TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE * 10
    print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
    print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)
    print("TEST_SEQ_LEN", TEST_SEQ_LEN)

    data = ParametricCaptureData(root_zarr_dir=opts.training_data_z)
    train_ds = data.tf_training_dataset(
        seq_len=TRAIN_SEQ_LEN,
        num_batches=opts.num_train_batches,
        batch_size=opts.batch_size,
        cache_fname=opts.cache_fname,
    )
    validate_ds = data.tf_training_dataset(
        seq_len=TEST_SEQ_LEN,
        num_batches=opts.num_validate_batches,
        batch_size=opts.batch_size,
    )

    # make model
    model_config = {
        "in_d": IN_D,
        "filter_sizes": FILTER_SIZES,
        "kernel_size": K,
        "out_d": OUT_D,
        "all_outputs": False,
    }
    with open("runs" / opts.run / "model_config.json", "w") as f:
        json.dump(model_config, f)
    train_model = create_dilated_model(**model_config)
    train_model.compile(Adam(opts.learning_rate), loss=masked_mse(RECEPTIVE_FIELD_SIZE))

    callbacks = []
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir)))
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_dir / "{epoch:03d}.weights.h5"),
            save_weights_only=True,
        )
    )
    callbacks.append(CheckYPred(tb_dir=str(tensorboard_dir), dataset=validate_ds))

    train_model.fit(train_ds, callbacks=callbacks, epochs=opts.epochs)
