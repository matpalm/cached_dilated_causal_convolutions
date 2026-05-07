import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import pickle, os, json, contextlib

# from tf_data_pipeline.data import WaveToWaveData, Embed2DWaveFormData
from tf_data_pipeline.interp_data import Embed2DInterpolatedWaveFormData

from qkeras.utils import model_save_quantized_weights

from .util import ensure_dir_exists, CheckYPred
from .qkeras_model import QKerasModelBuilder, masked_mse

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root-dir', type=str, required=True)
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--receptive-field-size",
        type=int,
        default=None,
        help="override RFS. if not set, use K^len(filter_sizes)",
    )
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument('--in-out-d', type=int, default=4)
    parser.add_argument(
        "--filter-sizes",
        type=int,
        nargs="+",
        required=True,
        help="sfeature depths for each layer; last layer always 4",
    )
    parser.add_argument('--po2-filter-size', type=int, default=None)
    parser.add_argument('--num-train-egs', type=int, default=200_000)
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
    opts = parser.parse_args()
    print("opts", opts)

    ensure_dir_exists(f"runs/{opts.run}/")
    for w in ["keras", "qkeras"]:
        ensure_dir_exists(f"runs/{opts.run}/weights/{w}")

    data = Embed2DInterpolatedWaveFormData(
        root_dir=opts.data_root_dir,
        pad_size=opts.in_out_d,
        seed=456,
    )

    # we only care about the loss of the _first_ element of the output
    # TODO: for amaranth version we should include a final OUT_D=1 layer
    filter_column_idx = 0

    # all convolutions use K=4
    K = 4
    num_layers = len(opts.filter_sizes) + 1

    # note: kernel size and implied dilation rate always assumed K
    RECEPTIVE_FIELD_SIZE = opts.receptive_field_size or K**num_layers
    TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE
    TRAIN_SEQ_LEN = RECEPTIVE_FIELD_SIZE * 5
    print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
    print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)
    print("TEST_SEQ_LEN", TEST_SEQ_LEN)

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
    train_ds = data.tf_dataset_for_split(
        "train", TRAIN_SEQ_LEN, opts.num_train_egs, interpolated_samples=True
    )
    validate_ds = data.tf_dataset_for_split(
        "validate", TRAIN_SEQ_LEN, opts.num_validate_egs, interpolated_samples=True
    )

    # construct some callbacks...

    # tensorboard
    tensorboard_dir = f"runs/{opts.run}/tb"
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

    # checkpointing raw keras weights
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"runs/{opts.run}/weights/keras/" + "{epoch:03d}",
        save_weights_only=True,
    )

    # plotting examples of validation data ( in tensorboard )
    check_y_pred_cb = CheckYPred(
        tb_dir=tensorboard_dir, dataset=validate_ds, model=train_model)

    # exporting qkeras quantised weights
    class SaveQuantisedWeights(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # save quantised weights dict pickled
            quantised_weights = model_save_quantized_weights(train_model)
            with open(f"runs/{opts.run}/weights/qkeras/e{epoch:02d}.pkl", 'wb') as f:
                pickle.dump(quantised_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
            #  add a latest symlink
            latest_symlink_fname = f"runs/{opts.run}/weights/qkeras/latest.pkl"
            try:
                os.remove(latest_symlink_fname)
            except FileNotFoundError:
                pass
            os.symlink(f"e{epoch:02d}.pkl", latest_symlink_fname)
    save_quantised_weights_cb = SaveQuantisedWeights()

    # def lr_schedule(epoch, lr):
    #     if epoch <= 40:
    #         print(epoch, "1e-4")
    #         return 1e-4
    #     else:
    #         print(epoch, "1e-5")
    #         return 1e-5
    # lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    # compile and train
    train_model.compile(Adam(opts.learning_rate),
                        loss=masked_mse(RECEPTIVE_FIELD_SIZE, filter_column_idx))
    train_model.fit(train_ds,
                    validation_data=validate_ds,
                    callbacks=[tensorboard_cb, checkpoint_cb,
                               check_y_pred_cb, save_quantised_weights_cb], #, lr_cb],
                    epochs=opts.epochs)
