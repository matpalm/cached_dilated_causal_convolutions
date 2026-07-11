import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import json
import contextlib
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# from tf_data_pipeline.data import WaveToWaveData, Embed2DWaveFormData
from tf_data_pipeline.quadrature_data import Embed2DQuadratureData
from qkeras.utils import model_save_quantized_weights

from .util import ensure_dir_exists, CheckYPred
from .qkeras_model import QKerasModelBuilder
from common.losses import combined_masked_loss_terms
<<<<<<< HEAD
from common.callbacks import setup_beta_stft_var_and_update_callback
=======
>>>>>>> master

import warnings

# suppress: UserWarning: The `keras.constraints.serialize()` API should only be used for objects of
#           type `keras.constraints.Constraint`. Found an instance of type
#           <class 'qkeras.quantizers.quantized_bits'>, which may lead to improper serialization.
warnings.filterwarnings(
    "ignore", category=UserWarning, message=r".*API should only be used for objects.*"
)

if __name__ == "__main__":

    # TODO: pathify run

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--receptive-field-size",
        type=int,
        default=None,
        help="override RFS. if not set, use K^len(filter_sizes)",
    )
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument(
        "--train-seq-len-multiplier",
        type=int,
        default=5,
        help="multiplier for receptive field to decide training sequence length.",
    )
    parser.add_argument("--in-d", type=int, default=4)
    parser.add_argument("--out-d", type=int, default=1)
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
        type=Path,
        default=None,
        help="path to keras weights used to initialise fine-tuning",
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
    parser.add_argument(
        "--train-interp",
        action="store_true",
        help="whether to train with interpolated samples",
    )
    parser.add_argument(
        "--double-interp",
        action="store_true",
        help="if set, and training with --train-interp, we interpolate e0 and e1 across sample",
    )
    opts = parser.parse_args()

    print("opts", opts)
    ensure_dir_exists(f"runs/{opts.run}/")
    for w in ["keras", "qkeras"]:
        ensure_dir_exists(f"runs/{opts.run}/weights/{w}")
    with open(f"runs/{opts.run}/opts.json", "w") as f:
        str_opts = {k: str(v) for k, v in vars(opts).items()}
        json.dump(str_opts, f)

    data = Embed2DQuadratureData(
        min_note=opts.min_note,
        max_note=opts.max_note,
        sample_rate_khz=opts.sample_rate_khz,
        fp_int=opts.fp_int,
        fp_frac=opts.fp_frac,
        harsh=opts.harsh_waves,
        soft_clip=opts.soft_clip,
        seed=456,
    )

    # all convolutions use K=4
    K = 4
    num_layers = len(opts.filter_sizes) + 1

    # note: kernel size and implied dilation rate always assumed K
    # receptive field should be _at least_ 128, even for 2 layer models otherwise we get no useful debug result
    RECEPTIVE_FIELD_SIZE = opts.receptive_field_size or K**num_layers
    RECEPTIVE_FIELD_SIZE = max(128, RECEPTIVE_FIELD_SIZE)
    TRAIN_SEQ_LEN = RECEPTIVE_FIELD_SIZE * opts.train_seq_len_multiplier
    print(
        f"RECEPTIVE_FIELD_SIZE {RECEPTIVE_FIELD_SIZE}"
        f" ( {opts.sample_rate_khz / RECEPTIVE_FIELD_SIZE} sec )",
    )
    print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)

    # construct model
    builder = QKerasModelBuilder(n_int=opts.fp_int, n_frac=opts.fp_frac)
    train_model = builder.create_dilated_model(
        TRAIN_SEQ_LEN,
        in_d=opts.in_d,
        out_d=opts.out_d,
        filter_sizes=opts.filter_sizes,
        # po2_filter_size=opts.po2_filter_size,  # if None, don't use po2
        l2=opts.l2,
        relu_upper_bound=opts.relu_upper_bound,
    )

    train_model.summary()

    if opts.init_weights and opts.init_weights.is_dir():
        init_weight_fname = sorted(os.listdir(opts.init_weights))[-1]
        init_weights_path = str(opts.init_weights / init_weight_fname)
        print("init weights from", init_weights_path)
        train_model.load_weights(init_weights_path)
    else:
        init_weights_path = None

    # make tf datasets
    train_ds = data.tf_dataset(
        batch_size=opts.batch_size,
        seq_len=TRAIN_SEQ_LEN,
        num_samples=opts.num_train_egs,
        emit_endpt_samples=True,
        emit_interpolated_samples=opts.train_interp,
        emit_double_interpolated_samples=opts.double_interp,
    )
    validate_ds = data.tf_dataset(
        batch_size=opts.batch_size,
        seq_len=TRAIN_SEQ_LEN,
        num_samples=opts.num_validate_egs,
        emit_endpt_samples=True,
        emit_interpolated_samples=opts.train_interp,
        emit_double_interpolated_samples=opts.double_interp,
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

    # If warm-up/ramp is requested, start from zero.

    ramp_callback, beta_stft = setup_beta_stft_var_and_update_callback(
        opts.epochs, opts.beta_stft_warmup, opts.beta_stft_ramp, opts.beta_stft
    )
    if ramp_callback is not None:
        callbacks.append(ramp_callback)

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
        use_huber_loss=opts.use_huber_loss,  # for now
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
        validation_data=None,  # just use validation for plots
        callbacks=callbacks,
        epochs=opts.epochs,
        verbose=2,
    )

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
                "init_weights_path": init_weights_path,
            },
            f,
        )
