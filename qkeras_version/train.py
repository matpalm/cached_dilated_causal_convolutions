
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import pickle, os

#from tf_data_pipeline.data import WaveToWaveData, Embed2DWaveFormData
from tf_data_pipeline.interp_data import Embed2DInterpolatedWaveFormData

from qkeras.utils import model_save_quantized_weights

from .util import ensure_dir_exists, CheckYPred
from .qkeras_model import create_dilated_model, masked_mse

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root-dir', type=str, required=True)
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--in-out-d', type=int, required=True)
    parser.add_argument('--filter-size', type=int, required=True)
    parser.add_argument('--num-train-egs', type=int, default=200_000)
    parser.add_argument('--num-validate-egs', type=int, default=10)
    parser.add_argument('--data-rescaling-factor', type=float, default=1.953125)
    #parser.add_argument('--root-weights-dir', type=str, default='weights')
    #parser.add_argument('--tb-dir', type=str, default='tb', help='tensorboard logs')
    opts = parser.parse_args()
    print("opts", opts)

    ensure_dir_exists(f"runs/{opts.run}/")
    for w in ['keras', 'qkeras', 'verilog']:
        ensure_dir_exists(f"runs/{opts.run}/weights/{w}")

    data = Embed2DInterpolatedWaveFormData(
        root_dir=opts.data_root_dir,
        rescaling_factor=opts.data_rescaling_factor,
        pad_size=opts.in_out_d,
        seed=456)

    # we only care about the loss of the _first_ element of the output
    filter_column_idx = 0

    # all convolutions use K=4
    K = 4

    # note: kernel size and implied dilation rate always assumed K
    RECEPTIVE_FIELD_SIZE = K**opts.num_layers
    TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE
    TRAIN_SEQ_LEN = RECEPTIVE_FIELD_SIZE * 10
    print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
    print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)
    print("TEST_SEQ_LEN", TEST_SEQ_LEN)

    # construct model
    train_model = create_dilated_model(TRAIN_SEQ_LEN,
            in_out_d=opts.in_out_d,
            num_layers=opts.num_layers,
            filter_size=opts.filter_size,
            l2=opts.l2,
            all_outputs=False)
    print(train_model.summary())

    # make tf datasets
    train_ds = data.tf_dataset_for_split('train', TRAIN_SEQ_LEN, opts.num_train_egs)
    validate_ds = data.tf_dataset_for_split('validate', TRAIN_SEQ_LEN, opts.num_validate_egs)

    # construct some callbacks...

    ## tensorboard
    tensorboard_dir = f"runs/{opts.run}/tb"
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

    ## checkpointing raw keras weights
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"runs/{opts.run}/weights/keras/" + "{epoch:03d}-{val_loss:.5f}",
        save_weights_only=True
    )

    ## plotting examples of validation data ( in tensorboard )
    check_y_pred_cb = CheckYPred(
        tb_dir=tensorboard_dir, dataset=validate_ds, model=train_model)

    ## exporting qkeras quantised weights
    class SaveQuantisedWeights(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            quantised_weights = model_save_quantized_weights(train_model)
            with open(f"runs/{opts.run}/weights/qkeras/e{epoch:02d}.pkl", 'wb') as f:
                pickle.dump(quantised_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_quantised_weights_cb = SaveQuantisedWeights()

    def lr_schedule(epoch, lr):
        if epoch <= 40:
            print(epoch, "1e-4")
            return 1e-4
        else:
            print(epoch, "1e-5")
            return 1e-5
    lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    # compile and train
    train_model.compile(Adam(opts.learning_rate),
                        loss=masked_mse(RECEPTIVE_FIELD_SIZE, filter_column_idx))
    train_model.fit(train_ds,
                    validation_data=validate_ds,
                    callbacks=[tensorboard_cb, checkpoint_cb,
                               check_y_pred_cb, save_quantised_weights_cb, lr_cb],
                    epochs=opts.epochs)

