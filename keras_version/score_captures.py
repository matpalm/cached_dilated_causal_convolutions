import argparse
import json
import math
from pathlib import Path
import zarr
import numpy as np
import pandas as pd
from tensorflow.keras.losses import MSE, Huber
from tqdm import tqdm

from .keras_model import create_dilated_model
from tf_data_pipeline.pcapture_data import ParametricCaptureData
from common.losses import masked_multires_stft_loss
from common.util import model_data_z_path_for
from common.sample_db import SampleDB

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--keras-run", type=Path, required=True)
parser.add_argument("--capture-run", type=Path, required=True)
parser.add_argument(
    "--alpha-mse",
    type=float,
    default=1.0,
    help="weight for huber in combined loss column",
)
parser.add_argument(
    "--beta-stft",
    type=float,
    default=0.1,
    help="weight for stft in combined loss column",
)
opts = parser.parse_args()
print("opts", opts)

# pcapture_data = ParametricCaptureData(capture_run=opts.capture_run)
# ds = pcapture_data.tf_inference_dataset(batch_size=8, return_sample_info=True)
# for d in ds:
#     print(d)
#     break
# exit()

db = SampleDB()

with open("runs" / opts.keras_run / "model_config.json", "r") as f:
    model_config = json.load(f)
    print("model_config", model_config)

inference_model = create_dilated_model(**model_config)

ckpts = (Path("runs") / opts.keras_run / "weights" / "keras").iterdir()
latest_ckpt = list(sorted(ckpts))[-1]
print("using ckpt", latest_ckpt)
inference_model.load_weights(str(latest_ckpt))

print(inference_model.summary())

# delta=0.1 reasonable for (-1, .1)
huber_loss = Huber(delta=0.1)
stft_loss = masked_multires_stft_loss()
pcapture_data = ParametricCaptureData(capture_run=opts.capture_run)
batch_size = 8
ds = pcapture_data.tf_inference_dataset(batch_size=batch_size, return_sample_info=True)
num_batches = math.ceil(pcapture_data.n_chunks / batch_size)
for x_b, y_true_b, _model_path_z_b, idx_b in tqdm(ds, total=num_batches):
    y_pred_b = inference_model(x_b)
    batch_n = y_true_b.shape[0]
    for i in range(batch_n):
        idx = int(idx_b[i])
        y_true = y_true_b[i : i + 1]
        y_pred = y_pred_b[i : i + 1]
        huber = float(huber_loss(y_true, y_pred))
        stft = float(stft_loss(y_true, y_pred))
        loss = (opts.alpha_mse * huber) + (opts.beta_stft * stft)
        db.set_losses(opts.capture_run, idx, opts.keras_run, loss, huber, stft)
