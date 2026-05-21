import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import pickle, os, json, contextlib

# from tf_data_pipeline.data import WaveToWaveData, Embed2DWaveFormData
from tf_data_pipeline.quadrature_data import Embed2DQuadratureData

from qkeras.utils import model_save_quantized_weights

from .util import ensure_dir_exists, CheckYPred
from .qkeras_model import QKerasModelBuilder
from .losses import combined_masked_loss_terms

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--receptive-field-size",
        type=int,
        default=None,
        help="override RFS. if not set, use K^len(filter_sizes)",
    )
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--in-out-d", type=int, default=4)
    parser.add_argument(
        "--filter-sizes",
        type=int,
        nargs="+",
        required=True,
        help="sfeature depths for each layer; last layer always 4",
    )
    # parser.add_argument("--po2-filter-size", type=int, default=None)
    parser.add_argument("--num-train-egs", type=int, default=200_000)
    parser.add_argument("--num-validate-egs", type=int, default=100)
    parser.add_argument("--fp-int", type=int, default=4)
    parser.add_argument("--fp-frac", type=int, default=12)
    parser.add_argument(
        "--init-weights",
        type=str,
        default=None,
        help="path to keras weights used to initialize fine-tuning",
    )
    parser.add_argument("--relu-upper-bound", type=float, default=6)
    parser.add_argument("--min-note", type=str, default="A2")
    parser.add_argument("--max-note", type=str, default="A4")
    parser.add_argument("--harsh-waves", action="store_true")
    parser.add_argument("--soft-clip", action="store_true")
    parser.add_argument("--sample-rate-khz", type=float, default=192)
    parser.add_argument(
        "--alpha-mse",
        type=float,
        default=1.0,
        help="weight for masked MSE in combined loss",
    )
    parser.add_argument(
        "--beta-stft",
        type=float,
        default=0.1,
        help="target STFT-loss weight after ramp",
    )
    parser.add_argument(
        "--beta-stft-ramp-epochs",
        type=int,
        default=0,
        help="linearly ramp beta_stft from 0 to target over this many epochs. 0 denotes no sftf",
    )
    parser.add_argument(
        "--train-interp",
        action="store_true",
        help="whether to train with interpolated samples",
    )
    opts = parser.parse_args()
    print("opts", opts)

    ensure_dir_exists(f"runs/{opts.run}/")
    for w in ["keras", "qkeras"]:
        ensure_dir_exists(f"runs/{opts.run}/weights/{w}")

    data = Embed2DQuadratureData(
        min_note=opts.min_note,
        max_note=opts.max_note,
        sample_rate_khz=opts.sample_rate_khz,
        harsh=opts.harsh_waves,
        soft_clip=opts.soft_clip,
        seed=456,
    )

    # we only care about the loss of the _first_ element of the output
    # TODO: for amaranth version we should include a final OUT_D=1 layer
    filter_column_idx = 0

    # all convolutions use K=4
    K = 4
    num_layers = len(opts.filter_sizes) + 1

    # note: kernel size and implied dilation rate always assumed K
    # receptive field should be _at least_ 128, even for 2 layer models otherwise we get no useful debug result
    RECEPTIVE_FIELD_SIZE = opts.receptive_field_size or K**num_layers
    RECEPTIVE_FIELD_SIZE = max(128, RECEPTIVE_FIELD_SIZE)
    TRAIN_SEQ_LEN = RECEPTIVE_FIELD_SIZE * 5
    print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
    print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)

    # construct model
    builder = QKerasModelBuilder(n_int=opts.fp_int, n_frac=opts.fp_frac)
    train_model = builder.create_dilated_model(
        TRAIN_SEQ_LEN,
        in_out_d=opts.in_out_d,
        filter_sizes=opts.filter_sizes,
        # po2_filter_size=opts.po2_filter_size,  # if None, don't use po2
        l2=opts.l2,
        relu_upper_bound=opts.relu_upper_bound,
    )

    if opts.init_weights is not None:
        print(f"loading initial weights from {opts.init_weights}")
        train_model.load_weights(opts.init_weights)

    train_model.summary()
    with open(f"runs/{opts.run}/qkeras_model.summary.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            train_model.summary()
    with open(f"runs/{opts.run}/qkeras_model.layer_info.json", "w") as f:
        json.dump(builder.layer_info, f)
    with open(f"runs/{opts.run}/qkeras_model.fp_config.json", "w") as f:
        json.dump(
            {
                "n_int": builder.n_int,
                "n_frac": builder.n_frac,
                "init_weights": opts.init_weights,
            },
            f,
        )

    # make tf datasets
    train_ds = data.tf_dataset(
        batch_size=64,
        seq_len=TRAIN_SEQ_LEN,
        num_samples=opts.num_train_egs,
        emit_endpt_samples=True,
        emit_interpolated_samples=opts.train_interp,
    )
    validate_ds = data.tf_dataset(
        batch_size=64,
        seq_len=TRAIN_SEQ_LEN,
        num_samples=opts.num_validate_egs,
        emit_endpt_samples=True,
        emit_interpolated_samples=opts.train_interp,
    )

    # construct some callbacks...
    callbacks = []

    # tensorboard
    tensorboard_dir = f"runs/{opts.run}/tb"
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)
    callbacks.append(tensorboard_cb)

    # checkpointing raw keras weights
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"runs/{opts.run}/weights/keras/" + "{epoch:03d}" + ".weights.h5",
            save_weights_only=True,
        )
    )

    # plotting examples of validation data ( in tensorboard )
    callbacks.append(CheckYPred(tb_dir=tensorboard_dir, dataset=validate_ds))

    # exporting qkeras quantised weights
    class SaveQuantisedWeights(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # save quantised weights dict pickled
            quantised_weights = model_save_quantized_weights(train_model)
            with open(f"runs/{opts.run}/weights/qkeras/e{epoch:02d}.pkl", "wb") as f:
                pickle.dump(quantised_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
            #  add a latest symlink
            latest_symlink_fname = f"runs/{opts.run}/weights/qkeras/latest.pkl"
            try:
                os.remove(latest_symlink_fname)
            except FileNotFoundError:
                pass
            os.symlink(f"e{epoch:02d}.pkl", latest_symlink_fname)
    callbacks.append(SaveQuantisedWeights())

    # beta_stft is captured by the loss closure and updated by callback.
    # stays at zero if no config around being updated by callback
    beta_stft = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    if opts.beta_stft_ramp_epochs > 0:

        class RampBetaStft(tf.keras.callbacks.Callback):
            def __init__(self, beta_var: tf.Variable, target: float, ramp_epochs: int):
                self.beta_var = beta_var
                self.target = float(target)
                self.ramp_epochs = max(1, int(ramp_epochs))

            def on_epoch_begin(self, epoch, logs=None):
                # epoch is 0-indexed; start at 0 and hit target at the end of ramp.
                if self.ramp_epochs == 1:
                    value = self.target
                else:
                    value = self.target * min(epoch / (self.ramp_epochs - 1), 1.0)
                self.beta_var.assign(value)
                print(f"epoch {epoch}: beta_stft={float(self.beta_var.numpy()):.6f}")

        callbacks.append(
            RampBetaStft(
                beta_var=beta_stft,
                target=opts.beta_stft,
                ramp_epochs=opts.beta_stft_ramp_epochs,
            )
        )

    # def lr_schedule(epoch, lr):
    #     if epoch <= 40:
    #         print(epoch, "1e-4")
    #         return 1e-4
    #     else:
    #         print(epoch, "1e-5")
    #         return 1e-5
    # lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    # compile and train
    combined_loss_fn, mse_loss_metric, stft_loss_metric = combined_masked_loss_terms(
        RECEPTIVE_FIELD_SIZE,
        filter_column_idx=filter_column_idx,
        alpha_mse=opts.alpha_mse,
        beta_stft=beta_stft,
    )

    train_model.compile(
        Adam(opts.learning_rate),
        loss=combined_loss_fn,
        metrics=[mse_loss_metric, stft_loss_metric],
    )
    train_model.fit(
        train_ds,
        validation_data=validate_ds,
        callbacks=callbacks,
        epochs=opts.epochs,
    )
