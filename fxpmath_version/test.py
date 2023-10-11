from fxpmath_version.fxpmath_model import FxpModel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tf_data_pipeline.data import WaveToWaveData

IN_OUT_D = 8
NUM_LAYERS = 3
FILTER_SIZE = 8

# note: kernel size and implied dilation rate always assumed 4
RECEPTIVE_FIELD_SIZE = 4**NUM_LAYERS
TEST_SEQ_LEN = RECEPTIVE_FIELD_SIZE
TRAIN_SEQ_LEN = RECEPTIVE_FIELD_SIZE * 5
print("RECEPTIVE_FIELD_SIZE", RECEPTIVE_FIELD_SIZE)
print("TRAIN_SEQ_LEN", TRAIN_SEQ_LEN)
print("TEST_SEQ_LEN", TEST_SEQ_LEN)
num_test_egs = 100

# make tf datasets
# recall WaveToWaveData
# x -> (tri,0,0,0)
# y -> (tri,square,zigzag,0)
data = WaveToWaveData()
# train_ds = data.tf_dataset_for_split('train', TRAIN_SEQ_LEN, opts.num_train_egs)
# validate_ds = data.tf_dataset_for_split('validate', TRAIN_SEQ_LEN, opts.num_validate_egs)
test_ds = data.tf_dataset_for_split('test', TEST_SEQ_LEN*5, num_test_egs)
for x, y in test_ds:
    break
x, y = x[0], y[0].numpy()

# run through fxp_model
fxp_model = FxpModel('qkeras_weights.pkl')

fxp_model.qconv0.export_weights_per_dot_product("/tmp/weights/qconv0")
fxp_model.qconv1.export_weights_per_dot_product("/tmp/weights/qconv1")
fxp_model.qconv2.export_weights_per_dot_product("/tmp/weights/qconv2")
fxp_model.qconv3.export_weights_per_dot_product("/tmp/weights/qconv3")

# run net
y_pred = []
for i in range(len(x)):
    if i%50 == 0: print(i, "/", len(x))
    y_pred.append(fxp_model.predict(x[i]))
y_pred = np.stack(y_pred)

# save numpy
np.save("y_pred.fxp_math.npy", y_pred)
np.save("test_y.npy", y)

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