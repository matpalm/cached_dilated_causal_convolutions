import numpy as np
import random
import tensorflow as tf
from pathlib import Path
import json
from collections import Counter
from tqdm import tqdm

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
    parser.add_argument(
        "--restore-run",
        type=Path,
        required=True,
        help="which run to restore checkpts from",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--sobol-capture-run",
        type=str,
        required=True,
        help="half of each batch is from this capture run; unbiased from sobol",
    )
    parser.add_argument(
        "--hard-capture-run",
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
    parser.add_argument("--train-batches-per-epoch", type=int, default=10_000)
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

    sobol_data = ParametricCaptureData(capture_run=opts.sobol_capture_run)
    sobol_train_ds = sobol_data.tf_training_dataset(
        seq_len=TRAIN_SEQ_LEN,
        num_batches=opts.train_batches_per_epoch,
        batch_size=opts.batch_size // 2,
        emit_idx=True,
    )
    print("sobol", opts.sobol_capture_run, "|egs|", sobol_data.num_examples())

    hard_data = ParametricCaptureImportanceSampledData(
        capture_run=opts.hard_capture_run, keras_model=opts.keras_model
    )
    print("hard_egs", opts.hard_capture_run, "|egs|", hard_data.num_examples())
    hard_train_ds = hard_data.tf_training_dataset(
        seq_len=TRAIN_SEQ_LEN,
        num_batches=opts.train_batches_per_epoch,
        batch_size=opts.batch_size // 2,
    )

    validate_ds = sobol_data.tf_training_dataset(
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

    # note: for importance sampling version we need the per element losses ( so we
    # don't reduce mean within the core loss fn )
    combined_loss_fn, mse_loss_metric, stft_loss_metric = combined_masked_loss_terms(
        RECEPTIVE_FIELD_SIZE,
        use_huber_loss=opts.use_huber_loss,
        alpha_mse=opts.alpha_mse,
        beta_stft=beta_stft,
        reduce_mean=False,
    )
    optimizer = Adam(opts.learning_rate)
    train_model.compile(
        optimizer,
        loss=combined_loss_fn,
        metrics=[mse_loss_metric, stft_loss_metric],
    )

    # with open("/dev/shm/sobol.losses.txt", "w") as f:
    #     for x_b, y_true_b in tqdm(
    #         sobol_data.tf_inference_dataset(batch_size=1),
    #         total=sobol_data.num_examples(),
    #         desc="sobol",
    #     ):
    #         y_pred_b = train_model(x_b)
    #         print(float(combined_loss_fn(y_true_b, y_pred_b)), file=f)
    #         f.flush()
    # with open("/dev/shm/is_egs.losses.txt", "w") as f:
    #     for x_b, y_true_b in tqdm(
    #         hard_data.tf_inference_dataset(batch_size=1),
    #         total=hard_data.num_examples(),
    #         desc="hard",
    #     ):
    #         y_pred_b = train_model(x_b)
    #         print(float(combined_loss_fn(y_true_b, y_pred_b)), file=f)
    #         f.flush()
    # exit()

    metric_core_name = getattr(mse_loss_metric, "__name__", "masked_mse")
    metric_stft_name = getattr(stft_loss_metric, "__name__", "masked_stft")

    callback_list = tf.keras.callbacks.CallbackList(
        callbacks,
        add_history=True,
        add_progbar=True,
        model=train_model,
        verbose=1,
        epochs=opts.epochs,
        steps=opts.train_batches_per_epoch,
    )
    callback_list.set_params(
        {
            "verbose": 1,
            "epochs": opts.epochs,
            "steps": opts.train_batches_per_epoch,
            "metrics": ["loss", metric_core_name, metric_stft_name],
        }
    )

    train_loss = tf.keras.metrics.Mean(name="loss")
    train_core = tf.keras.metrics.Mean(name=metric_core_name)
    train_stft = tf.keras.metrics.Mean(name=metric_stft_name)

    @tf.function
    def train_step(x_b, y_b, weight_b):

        # calculate per element loss and multiply by weights from priority replay
        with tf.GradientTape() as tape:
            y_pred = train_model(x_b, training=True)
            per_element_loss_values = combined_loss_fn(y_b, y_pred)
            loss_value = tf.reduce_mean(per_element_loss_values * weight_b)

        gradients = tape.gradient(loss_value, train_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, train_model.trainable_variables))

        core_value = tf.reduce_mean(mse_loss_metric(y_b, y_pred))
        stft_value = tf.reduce_mean(stft_loss_metric(y_b, y_pred))

        return loss_value, core_value, stft_value, per_element_loss_values

    callback_list.on_train_begin()

    loss_log = open("loss_log.tsv", "w")
    print(
        "\t".join("epoch step batch_idx eg_type idx freq weight loss".split(" ")),
        file=loss_log,
    )

    # track number of times each of the sobol and IS samples
    # have been sampled. just for loss.tsv debugging
    times_sampled_sobol = Counter()
    times_sampled_is = Counter()

    for epoch in range(opts.epochs):
        # print(">>>EPOCH", epoch)
        callback_list.on_epoch_begin(epoch)

        train_loss.reset_state()
        train_core.reset_state()
        train_stft.reset_state()

        sobol_train_iter = iter(sobol_train_ds)
        hard_train_iter = iter(hard_train_ds)

        # christ; why didn't i write this in jax :/

        try:
            step = 0
            while True:
                # print(">>>STEP", step)

                def le(t):
                    return list(enumerate(t.numpy().tolist()))

                callback_list.on_train_batch_begin(step)

                # the sobol and hard examples provide half the batch each
                x_sobol_b, y_sobol_b, idx_sobol_b = next(sobol_train_iter)
                x_hard_b, y_hard_b, idx_is_b, weight_is_b = next(hard_train_iter)

                # update debug sample counts  DEBUG
                idx_sobol_b = idx_sobol_b.numpy().tolist()
                idx_is_b = idx_is_b.numpy().tolist()
                times_sampled_sobol.update(idx_sobol_b)
                times_sampled_is.update(idx_is_b)

                # print("idx_sobol_b", list(enumerate(idx_sobol_b)))
                # print("is_idx_b", list(enumerate(idx_is_b)))
                # print("weight_hard_b", le(weight_is_b))

                # use a fixed weight for the sobol samples
                weight_sobol_b = tf.ones([tf.shape(x_sobol_b)[0]])

                # vstack both into a batch
                x_b = tf.concat([x_sobol_b, x_hard_b], axis=0)
                y_b = tf.concat([y_sobol_b, y_hard_b], axis=0)
                weight_b = tf.concat([weight_sobol_b, weight_is_b], axis=0)
                # print("combined weight_b", le(weight_b))
                # now that sobol and hard weights have been mixed we can
                # max normalise over the true batch
                weight_b = weight_b / tf.reduce_max(weight_b)
                # print("combined weight_b", le(weight_b))

                # run train step
                loss_value, core_value, stft_value, per_element_loss_b = train_step(
                    x_b, y_b, weight_b
                )

                print(
                    "loss_value",
                    loss_value,
                    "core_value",
                    core_value,
                    "stft_value",
                    stft_value,
                )
                per_element_loss_b = per_element_loss_b.numpy().tolist()

                # print("per_element_loss_b", list(enumerate(per_element_loss_b)))

                # extract the losses for the hard example from the second half of
                # the batch to reupdate prios
                n_sobol = tf.shape(x_sobol_b)[0]
                loss_is_b = per_element_loss_b[n_sobol:]

                # >>>> debug

                # update per element losses in sum tree
                hard_data.prio_replay.update(idx_is_b, loss_is_b)

                # print(
                #     "idx_is_b/loss_is_b",
                #     list(enumerate(zip(idx_is_b, loss_is_b))),
                # )

                # update loss_log.tsv
                #  epoch step batch_idx eg_type idx freq weight loss
                for b_idx in range(opts.batch_size):
                    if b_idx < n_sobol:
                        # sobol sample
                        eg_type = "s"
                        idx = idx_sobol_b[b_idx]
                        freq = times_sampled_sobol[idx]
                    else:
                        # IS sample
                        eg_type = "is"
                        idx = idx_is_b[b_idx - n_sobol]
                        freq = times_sampled_is[idx]
                    w = float(weight_b[b_idx])
                    l = float(per_element_loss_b[b_idx])
                    print(
                        "\t".join(
                            map(str, [epoch, step, b_idx, eg_type, idx, freq, w, l])
                        ),
                        file=loss_log,
                    )
                loss_log.flush()

                # <<<

                train_loss.update_state(loss_value)
                train_core.update_state(core_value)
                train_stft.update_state(stft_value)

                batch_logs = {
                    "loss": float(train_loss.result().numpy()),
                    metric_core_name: float(train_core.result().numpy()),
                    metric_stft_name: float(train_stft.result().numpy()),
                }

                callback_list.on_train_batch_end(step, batch_logs)

                step += 1

        except StopIteration as sie:
            # end of epoch
            pass

        hard_data.prio_replay.dump(epoch)

        epoch_logs = {
            "loss": float(train_loss.result().numpy()),
            metric_core_name: float(train_core.result().numpy()),
            metric_stft_name: float(train_stft.result().numpy()),
        }

        callback_list.on_epoch_end(epoch, epoch_logs)

    callback_list.on_train_end()
