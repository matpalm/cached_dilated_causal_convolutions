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

# from tf_data_pipeline.data import Embed2DWaveFormData
from tf_data_pipeline.pcapture_data import ParametricCaptureData
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
        "--capture-run",
        type=str,
        required=True,
        help="capture run for training data zarr",
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

    print("tf", tf.__version__)
    print("tf devices", tf.config.list_physical_devices())

    tensorboard_dir = "runs" / opts.run / "tb"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = "runs" / opts.run / "weights" / "keras"
    weights_dir.mkdir(parents=True, exist_ok=True)

    IN_D = 4  # triangle, 3 cvs
    OUT_D = 1  # output wave
    K = 4
    FILTER_SIZES = [16, 32, 64, 128, 128]
    RECEPTIVE_FIELD_SIZE = K ** len(FILTER_SIZES)
    TRAIN_SEQ_LEN = RECEPTIVE_FIELD_SIZE * 5
    TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE * 10
    print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
    print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)
    print("TEST_SEQ_LEN", TEST_SEQ_LEN)

    data = ParametricCaptureData(capture_run=opts.capture_run)
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
        "skip_dim": 32,
        "out_d": OUT_D,
        "all_outputs": False,
    }
    print("model_config", model_config)
    with open("runs" / opts.run / "model_config.json", "w") as f:
        json.dump(model_config, f)
    train_model = create_dilated_model(**model_config)
    train_model.summary()

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
        jit_compile=False,  # XLA problem with STFT ???
    )

    def run_fit_loop():
        train_model.fit(train_ds, callbacks=callbacks, epochs=opts.epochs)
        # train_model.fit(train_ds, callbacks=[], epochs=opts.epochs)

    # def run_custom():

    #     @tf.function
    #     def train_step(x_b, y_b):
    #         with tf.GradientTape() as tape:
    #             y_pred = train_model(x_b, training=True)
    #             loss = combined_loss_fn(y_b, y_pred)
    #         gradients = tape.gradient(loss, train_model.trainable_variables)
    #         optimizer.apply_gradients(zip(gradients, train_model.trainable_variables))
    #         return loss

    #     fetch_ts = []
    #     step_ts = []
    #     last_t = overall_start_t = time.perf_counter()
    #     for i, (xs_b, ys_b) in enumerate(train_ds):
    #         print(xs_b.shape)
    #         fetch_t = time.perf_counter() - last_t  # data time (the real ~8s)
    #         fetch_ts.append(fetch_t)
    #         start_t = time.perf_counter()
    #         loss = train_step(xs_b, ys_b)
    #         step_t = time.perf_counter() - start_t  # compute (~0.002s)
    #         step_ts.append(step_t)
    #         print(i, "fetch", fetch_t, "step", step_t)
    #         last_t = time.perf_counter()
    #     total_t = time.perf_counter() - overall_start_t
    #     print("total_t", total_t)

    #     print("fetch_ts mean", np.mean(fetch_ts), "std", np.std(fetch_ts))
    #     print("step_ts mean", np.mean(step_ts), "std", np.std(step_ts))

    run_fit_loop()
    # run_custom()
