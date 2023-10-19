0
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import pickle

from tf_data_pipeline.data import WaveToWaveData, Embed2DWaveFormData

from qkeras.utils import model_save_quantized_weights

from .qkeras_model import create_dilated_model, masked_mse

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--dataset', type=str, help='which dataset to train on')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--num-train-egs', type=int, default=200_000)
    parser.add_argument('--num-validate-egs', type=int, default=1_000)
    parser.add_argument('--data-rescaling-factor', type=float, default=1.953125)
    parser.add_argument('--save-weights', type=str, default='qkeras_weights')
    opts = parser.parse_args()
    print("opts", opts)

    # TODO: just focussing on Embed2DWaveFormData for now

    filter_column_idx = None
    # if opts.dataset == 'wave_to_wave':
    #     data = WaveToWaveData(
    #         root_dir='datalogger_firmware/data/2d_embed/32kHz',
    #         rescaling_factor=opts.data_rescaling_factor)
    #     filter_column_idx = 1  # square wave
    # elif opts.dataset == 'embed_2d':
    data = Embed2DWaveFormData(
        root_dir='datalogger_firmware/data/2d_embed/32kHz',
        rescaling_factor=opts.data_rescaling_factor)
    filter_column_idx = 0
    # else:
    #     raise Exception("unknown --dataset")

    K = 4
    IN_OUT_D = FILTER_SIZE = 8

    # note: kernel size and implied dilation rate always assumed K

    RECEPTIVE_FIELD_SIZE = K**opts.num_layers
    TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE
    TRAIN_SEQ_LEN = RECEPTIVE_FIELD_SIZE * 5
    print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
    print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)
    print("TEST_SEQ_LEN", TEST_SEQ_LEN)

    # make model
    train_model = create_dilated_model(TRAIN_SEQ_LEN,
            in_out_d=IN_OUT_D, num_layers=opts.num_layers,
            filter_size=FILTER_SIZE, l2=opts.l2,
            all_outputs=False)
    print(train_model.summary())

    # make tf datasets
    train_ds = data.tf_dataset_for_split('train', TRAIN_SEQ_LEN, opts.num_train_egs)
    validate_ds = data.tf_dataset_for_split('validate', TRAIN_SEQ_LEN, opts.num_validate_egs)

    # train model
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='keras_weights/{epoch:03d}-{val_loss:.5f}',
        save_weights_only=True
    )

    class SaveQuantisedWeights(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            quantised_weights = model_save_quantized_weights(train_model)
            with open(f"{opts.save_weights}_e{epoch:02d}.pkl", 'wb') as f:
                pickle.dump(quantised_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_quantised_weights_cb = SaveQuantisedWeights()

    train_model.compile(Adam(opts.learning_rate),
                        loss=masked_mse(RECEPTIVE_FIELD_SIZE, filter_column_idx))
    train_model.fit(train_ds,
                    validation_data=validate_ds,
                    callbacks=[checkpoint_cb, save_quantised_weights_cb],
                    epochs=opts.epochs)

