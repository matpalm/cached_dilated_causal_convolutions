
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import pickle

from tf_data_pipeline.data import WaveToWaveData

from qkeras.utils import model_save_quantized_weights

from .qkeras_model import create_dilated_model, masked_mse

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-train-egs', type=int, default=200_000)
    parser.add_argument('--num-validate-egs', type=int, default=1_000)
    parser.add_argument('--data-rescaling-factor', type=float, default=1.0)
    parser.add_argument('--save-weights', type=str, default='qkeras_weights.pkl')
    opts = parser.parse_args()
    print("opts", opts)

    # parse files and do splits etc
    data = WaveToWaveData(rescaling_factor=opts.data_rescaling_factor)

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
            filter_size=FILTER_SIZE,
            all_outputs=False)
    print(train_model.summary())

    # make tf datasets
    train_ds = data.tf_dataset_for_split('train', TRAIN_SEQ_LEN, opts.num_train_egs)
    validate_ds = data.tf_dataset_for_split('validate', TRAIN_SEQ_LEN, opts.num_validate_egs)

    # train model
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='weights/{epoch:03d}-{val_loss:.5f}',
        save_weights_only=True
    )
    train_model.compile(Adam(opts.learning_rate),
                        loss=masked_mse(RECEPTIVE_FIELD_SIZE, filter_column_idx=1))  # only calculate loss for col1, square wave
    train_model.fit(train_ds,
                    validation_data=validate_ds,
                    callbacks=[checkpoint_cb],
                    epochs=opts.epochs)

    quantised_weights = model_save_quantized_weights(train_model)

    with open(opts.save_weights, 'wb') as f:
        pickle.dump(quantised_weights, f, protocol=pickle.HIGHEST_PROTOCOL)


