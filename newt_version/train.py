import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from keras.losses import MSE, Huber
from tf_data_pipeline.pcapture_data import ParametricCaptureData
from .models import build_newt

# from .losses import masked_mse

from .util import CheckYPred

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--num-train-egs", type=int, default=100)
parser.add_argument("--num-epochs", type=int, default=10)
opts = parser.parse_args()
print("opts", opts)

data = ParametricCaptureData(root_zarr_dir="/dev/shm/r001/")

TRAIN_SEQ_LEN = 512
train_ds = data.tf_dataset(
    batch_size=opts.batch_size,
    seq_len=TRAIN_SEQ_LEN,
    num_batches=opts.num_train_egs,
    # emit_endpt_samples=True,
    # emit_interpolated_samples=opts.train_interp,
    # emit_double_interpolated_samples=opts.double_interp,
)

train_model = build_newt()
print(train_model.summary())

train_model.compile(
    Adam(opts.learning_rate),
    loss=Huber(),
)

check_ypred = CheckYPred(
    tb_dir="tb/newt",
    dataset=data.tf_dataset(
        batch_size=opts.batch_size,
        seq_len=TRAIN_SEQ_LEN * 5,
        num_batches=1,
    ),
)

train_model.fit(train_ds, epochs=opts.num_epochs, callbacks=[check_ypred])
