from fxpmath_version.fxpmath_model import FxpModel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tf_data_pipeline.data import WaveToWaveData

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-rescaling-factor', type=float, default=1.0)
parser.add_argument('--load-weights', type=str, default='qkeras_weights.pkl')
parser.add_argument('--num-test-egs', type=int, default=100)
opts = parser.parse_args()
print("opts", opts)

# run through fxp_model
fxp_model = FxpModel(opts.load_weights)

# export weights to tmp for verilog version
for i, qconv_layer in enumerate(fxp_model.qconvs):
    fname = f"weights/qconv{i}"
    print("exporting qconv", i, "to", fname)
    qconv_layer.export_weights_for_verilog(fname)

IN_OUT_D = FILTER_SIZE = 8

print("|layers|=", fxp_model.num_layers)
K = 4
RECEPTIVE_FIELD_SIZE = K**fxp_model.num_layers
TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE
print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
print("TEST_SEQ_LEN", TEST_SEQ_LEN)

# make tf datasets
# recall WaveToWaveData
# x -> (tri,0,0,0,...)
# y -> (tri,square,zigzag,0,0,...)
data = WaveToWaveData(rescaling_factor=opts.data_rescaling_factor)
test_ds = data.tf_dataset_for_split('test', opts.num_test_egs, opts.num_test_egs)
for x, y in test_ds:
    x, y = x[0], y[0].numpy()
    print("x", x.shape, "y", y.shape)
    IN_OUT_D = x.shape[1]
    assert x.shape == (opts.num_test_egs, IN_OUT_D), x.shape
    assert y.shape == (opts.num_test_egs, IN_OUT_D), y.shape
    break

# run net
y_pred = []
for i in range(len(x)):
    print("--------------------", i, "/", len(x))
    y_pred.append(fxp_model.predict(x[i]))
y_pred = np.stack(y_pred)

# save numpy
# np.save("y_pred.fxp_math.npy", y_pred)
# np.save("test_y.npy", y)

# save plot
df = pd.DataFrame()
df['y_pred'] = y_pred[:,1]     # recall; loss only covered element1
df['y_true'] = y[:,1]
df['n'] = range(len(y_pred))
wide_df = pd.melt(df, id_vars=['n'], value_vars=['y_pred', 'y_true'])
sns.lineplot(wide_df, x='n', y='value', hue='variable')
plt_fname = "fxp_math.y_pred.png"
print("saving plot to", plt_fname)
plt.savefig(plt_fname)