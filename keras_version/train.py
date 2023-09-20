import pandas as pd
import numpy as np
import random
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam

from .keras_model import create_dilated_model
from cmsisdsp_py_version.cached_block_model import create_cached_block_model_from_keras_model

from .data import WaveFormData

def masked_mse(receptive_field_size):
    def loss_fn(y_true, y_pred):
        assert len(y_true.shape) == 3, "expected (batch, sequence_length, output_dim)"
        assert y_true.shape == y_pred.shape
        # average over elements of y
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        # we want to ignore the first elements of the loss since they
        # have been fed with left padded data
        mse = mse[:,receptive_field_size:]
        # return average over batch and sequence
        return tf.reduce_mean(mse)
    return loss_fn

def wave_coords(wave):
    return {'sine': '(0, 0)', 'ramp': '(0, 1)',
            'square': '(1, 0)', 'zigzag': '(1, 1)' }[wave]

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num-train-egs', type=int, default=200_000)
    parser.add_argument('--num-validate-egs', type=int, default=1_000)
    opts = parser.parse_args()
    print("opts", opts)

    # parse files and do splits etc
    data = WaveFormData()

    # training config
    # prep training config
    IN_D = 3    # 2d embedding, (0,1) and core triangle
    OUT_D = 1   # output wave
    # kernel size and implied dilation rate
    K = 4
    # filters for Nth layer Kx1 and 1x1 convs
    # [4, 3, 8, 8] @ 32kHz => 72%
    # [4, 8, 8, 8] @ 32kHz => 82%
    # [4, 8, 8, 12] @ 32kHz => 93%
    # [8, 8, 8, 8] @ 32kHz => TOO MUCH
    # [4, 4, 4] @ 96kHz => too much :/
    # [2, 2, 4] @ 96kHz => too much :/
    # [2, 2, 2] @ 96kHz => too much :/
    FILTER_SIZES = [4, 4, 4, 8]
    RECEPTIVE_FIELD_SIZE = K**len(FILTER_SIZES)
    TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE
    TRAIN_SEQ_LEN = RECEPTIVE_FIELD_SIZE * 5
    print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
    print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)
    print("TEST_SEQ_LEN", TEST_SEQ_LEN)

    # make model
    train_model = create_dilated_model(
            TRAIN_SEQ_LEN, in_d=IN_D, filter_sizes=FILTER_SIZES,
            kernel_size=K, out_d=OUT_D,
            all_outputs=False)

    # make tf dataset
    train_ds, validate_ds = data.train_validate_tf_datasets(
        TRAIN_SEQ_LEN, opts.num_train_egs, opts.num_validate_egs)

    # train model
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath='weights/{epoch:03d}-{val_loss:.5f}',
        save_weights_only=True
    )

    train_model.compile(Adam(opts.learning_rate),
                        loss=masked_mse(RECEPTIVE_FIELD_SIZE))
    train_model.fit(train_ds,
                    validation_data=validate_ds,
                    callbacks=[checkpoint_cb],
                    epochs=opts.epochs)

    # generate graphs of y_pred against test data
    for wave in ['sine', 'ramp', 'square', 'zigzag']:

        test_records = []
        for i in range(1000, 1500):
            test_seq = data.tri_to[wave]['test']['x'][i:i+TRAIN_SEQ_LEN]
            test_seq = np.expand_dims(test_seq, 0)  # single element batch

            y_true = data.tri_to[wave]['test']['y'][i+TRAIN_SEQ_LEN]

            y_pred = train_model(test_seq).numpy()
            y_pred = y_pred[0,-1,:]  # train model gives all steps, we just want last

            for out_c in range(1):
                test_records.append((i, out_c, 'x', test_seq[0,-1,2]))
                test_records.append((i, out_c, 'y_true', y_true[out_c]))
                test_records.append((i, out_c, 'y_pred', y_pred[out_c]))

        test_df = pd.DataFrame(test_records, columns=['n', 'c', 'name', 'value'])

        plt.clf()
        plt.figure(figsize=(8, 3))
        ax = sns.lineplot(data=test_df[test_df['c']==0], x='n', y='value', hue='name')
        ax.set(title=f"test performance on {wave} {wave_coords(wave)}")
        ax.set_ylim(-1, 1)
        plt.savefig(f"/tmp/test_{wave}.png")

    # export model_defn.c
    cached_block_model = create_cached_block_model_from_keras_model(
        train_model, input_feature_depth=IN_D)
    with open("/tmp/model_defn.h", 'w') as f:
        cached_block_model.write_model_defn_h(f) #sys.stdout)
