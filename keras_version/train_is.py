import pandas as pd
import numpy as np
import random
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json

from tensorflow.keras.optimizers import Adam

from .keras_model import create_dilated_model

# from cmsisdsp_py_version.cached_block_model import create_cached_block_model_from_keras_model

from tf_data_pipeline.pcapture_data import ParametricCaptureData
from tf_data_pipeline.pcapture_is_data import ParametricCaptureImportanceSampledData

from .util import CheckYPred
from common.losses import combined_masked_loss_terms
from common.callbacks import setup_beta_stft_var_and_update_callback

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--unbiased-capture-run",
        type=str,
        required=True,
        help="half of each batch is from this capture run; unbiased from sobol",
    )
    parser.add_argument(
        "--biased-capture-run",
        type=str,
        required=True,
        help="half of each batch is from this capture run; sampled with importance sampling",
    )
    parser.add_argument(
        "--keras-model",
        type=str,
        required=True,
        help="the keras model losses to use from db for importance sampling",
    )
    parser.add_argument("--num-train-batches", type=int, default=10_000)
    parser.add_argument("--num-validate-batches", type=int, default=10)
    parser.add_argument("--cache-fname", type=str, default=None)
    parser.add_argument(
        "--alpha-mse",
        type=float,
        default=1.0,
        help="weight for masked MSE/Huber in combined loss",
    )
    parser.add_argument(
        "--use-huber-loss",
        action="store_true",
        help="if set use huber instead of MSE",
    )
    parser.add_argument(
        "--beta-stft",
        type=float,
        default=0.1,
        help="target STFT-loss weight after warm up and ramp",
    )
    parser.add_argument(
        "--beta-stft-warmup",
        type=float,
        default=0,
        help="keep beta_stft at 0 for this proportion of epochs at start",
    )
    parser.add_argument(
        "--beta-stft-ramp",
        type=float,
        default=0,
        help="linearly ramp beta_stft from 0 to target over this many proportion of epochs ( post warmup )",
    )
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

    unbiased_data = ParametricCaptureData(capture_run=opts.unbiased_capture_run)

    biased_data = ParametricCaptureImportanceSampledData(
        capture_run=opts.biased_capture_run, keras_model=opts.keras_model
    )

    train_ds = biased_data.tf_training_dataset(
        seq_len=TRAIN_SEQ_LEN,
        num_batches=opts.num_train_batches,
        batch_size=opts.batch_size,
    )
    validate_ds = unbiased_data.tf_training_dataset(
        seq_len=TEST_SEQ_LEN,
        num_batches=opts.num_validate_batches,
        batch_size=opts.batch_size,
    )

    # make model
    model_config = {
        "in_d": IN_D,
        "filter_sizes": FILTER_SIZES,
        "kernel_size": K,
        "skip_dim": 32,
        "out_d": OUT_D,
        "all_outputs": False,
    }
    with open("runs" / opts.run / "model_config.json", "w") as f:
        json.dump(model_config, f)
    train_model = create_dilated_model(**model_config)

    callbacks = []
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir)))
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_dir / "{epoch:03d}.weights.h5"),
            save_weights_only=True,
        )
    )
    callbacks.append(CheckYPred(tb_dir=str(tensorboard_dir), dataset=validate_ds))

    ramp_callback, beta_stft = setup_beta_stft_var_and_update_callback(
        opts.epochs, opts.beta_stft_warmup, opts.beta_stft_ramp, opts.beta_stft
    )
    if ramp_callback is not None:
        callbacks.append(ramp_callback)

    combined_loss_fn, mse_loss_metric, stft_loss_metric = combined_masked_loss_terms(
        RECEPTIVE_FIELD_SIZE,
        use_huber_loss=opts.use_huber_loss,
        alpha_mse=opts.alpha_mse,
        beta_stft=beta_stft,
    )
    optimizer = Adam(opts.learning_rate)
    train_model.compile(
        optimizer,
        loss=combined_loss_fn,
        metrics=[mse_loss_metric, stft_loss_metric],
    )

    metric_core_name = getattr(mse_loss_metric, "__name__", "masked_mse")
    metric_stft_name = getattr(stft_loss_metric, "__name__", "masked_stft")

    callback_list = tf.keras.callbacks.CallbackList(
        callbacks,
        add_history=True,
        add_progbar=True,
        model=train_model,
        verbose=1,
        epochs=opts.epochs,
        steps=opts.num_train_batches,
    )
    callback_list.set_params(
        {
            "verbose": 1,
            "epochs": opts.epochs,
            "steps": opts.num_train_batches,
            "metrics": ["loss", metric_core_name, metric_stft_name],
        }
    )

    train_loss = tf.keras.metrics.Mean(name="loss")
    train_core = tf.keras.metrics.Mean(name=metric_core_name)
    train_stft = tf.keras.metrics.Mean(name=metric_stft_name)

    @tf.function
    def train_step(x_b, y_b, weight_b):
        with tf.GradientTape() as tape:
            y_pred = train_model(x_b, training=True)
            loss_value = combined_loss_fn(y_b, y_pred)
        gradients = tape.gradient(loss_value, train_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, train_model.trainable_variables))

        core_value = mse_loss_metric(y_b, y_pred)
        stft_value = stft_loss_metric(y_b, y_pred)
        return loss_value, core_value, stft_value

    callback_list.on_train_begin()
    for epoch in range(opts.epochs):
        callback_list.on_epoch_begin(epoch)
        train_loss.reset_state()
        train_core.reset_state()
        train_stft.reset_state()

        for step, (x_b, y_b, idx_b, weight_b) in enumerate(train_ds):

            print("-" * 100)
            print("--STEP", step)
            print("idx_b", idx_b)
            print("weight_b", weight_b)

            callback_list.on_train_batch_begin(step)
            loss_value, core_value, stft_value = train_step(x_b, y_b, weight_b)

            train_loss.update_state(loss_value)
            train_core.update_state(core_value)
            train_stft.update_state(stft_value)

            batch_logs = {
                "loss": float(train_loss.result().numpy()),
                metric_core_name: float(train_core.result().numpy()),
                metric_stft_name: float(train_stft.result().numpy()),
            }
            callback_list.on_train_batch_end(step, batch_logs)

        epoch_logs = {
            "loss": float(train_loss.result().numpy()),
            metric_core_name: float(train_core.result().numpy()),
            metric_stft_name: float(train_stft.result().numpy()),
        }
        callback_list.on_epoch_end(epoch, epoch_logs)

    callback_list.on_train_end()
