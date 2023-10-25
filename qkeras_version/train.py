
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import pickle, os
import warnings

#from tf_data_pipeline.data import WaveToWaveData, Embed2DWaveFormData
from tf_data_pipeline.interp_data import Embed2DInterpolatedWaveFormData

from qkeras.utils import model_save_quantized_weights

import util

from .qkeras_model import create_dilated_model, masked_mse

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root-dir', type=str, required=True)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--in-out-d', type=int, required=True)
    parser.add_argument('--filter-size', type=int, required=True)
    parser.add_argument('--num-train-egs', type=int, default=200_000)
    parser.add_argument('--num-validate-egs', type=int, default=10)
    parser.add_argument('--data-rescaling-factor', type=float, default=1.953125)
    parser.add_argument('--root-weights-dir', type=str, default='weights')
    opts = parser.parse_args()
    print("opts", opts)

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

    # 1) a callback for checkpointing raw keras weights
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=opts.root_weights_dir+'/keras/{epoch:03d}-{val_loss:.5f}',
        save_weights_only=True
    )

    # 2) a callback to plot the result from a validation sample on epoch end
    # TODO: this code has been cutnpaste elsewhere so could move into a util
    class CheckYPred(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            for tx, ty in validate_ds:
                break
            y_pred = train_model(tx)
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            for i in range(20): #len(tx)):
                df = pd.DataFrame()
                df['x'] = tx[i,:,0]
                df['e0'] = tx[i,:,1]
                df['e1'] = tx[i,:,2]
                df['y_true'] = ty[i,:,0]
                df['y_pred'] = y_pred[i,:,0]
                df['n'] = range(len(tx[i]))
                wide_df = pd.melt(df, id_vars=['n'], value_vars=['x', 'y_pred', 'y_true', 'e0', 'e1'])
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore', category=FutureWarning)
                    #plt.figure(figsize=(20, 10))
                    p = sns.lineplot(wide_df, x='n', y='value', hue='variable')
                    p.set_ylim((-2, 2))
                    d = f"check_y_pred_cb/e{epoch:03d}"
                    util.ensure_dir_exists(d)
                    plt_fname = f"{d}/i{i:03d}.e0_{tx[i,0,1]:0.2f}_e1_{tx[i,0,2]:0.2f}.png"
                    plt.savefig(plt_fname)
                    plt.clf()
    check_y_pred_cb = CheckYPred()

    # 3) a callback to export qkeras quantised weights
    class SaveQuantisedWeights(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            quantised_weights = model_save_quantized_weights(train_model)
            util.ensure_dir_exists(opts.root_weights_dir+"/qkeras")
            with open(f"{opts.root_weights_dir}/qkeras/e{epoch:02d}.pkl", 'wb') as f:
                pickle.dump(quantised_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_quantised_weights_cb = SaveQuantisedWeights()

    # compile and train
    train_model.compile(Adam(opts.learning_rate),
                        loss=masked_mse(RECEPTIVE_FIELD_SIZE, filter_column_idx))
    train_model.fit(train_ds,
                    validation_data=validate_ds,
                    callbacks=[checkpoint_cb, save_quantised_weights_cb, check_y_pred_cb],
                    epochs=opts.epochs)

