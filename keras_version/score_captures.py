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
from common.losses import combined_masked_loss_terms
from common.sample_db import SampleDB

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--keras-run", type=Path, required=True)
parser.add_argument("--capture-runs", type=Path, required=True, nargs="+")
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

print("WARNING: assuming use_huber_loss=True")
_combined_loss_fn, mse_loss_fn, stft_loss_fn = combined_masked_loss_terms(
    receptive_field_size=None,
    use_huber_loss=True,
    alpha_mse=opts.alpha_mse,
    beta_stft=opts.beta_stft,
    reduce_mean=True,
)


# delta=0.1 reasonable for (-1, .1)
batch_size = 8
for capture_run in opts.capture_runs:
    pcapture_data = ParametricCaptureData(capture_run)
    ds = pcapture_data.tf_inference_dataset(
        batch_size=batch_size, return_sample_info=True
    )
    num_batches = math.ceil(pcapture_data.n_chunks / batch_size)
    for x_b, y_true_b, _model_path_z_b, idx_b in tqdm(ds, total=num_batches):
        y_pred_b = inference_model(x_b)
        batch_n = y_true_b.shape[0]
        for i in range(batch_n):
            idx = int(idx_b[i])
            y_true = y_true_b[i : i + 1]
            y_pred = y_pred_b[i : i + 1]
            huber = float(mse_loss_fn(y_true, y_pred))
            stft = float(stft_loss_fn(y_true, y_pred))
            # this is slightly dangerous, since if combined_loss_fn changes
            # we are testing the wrong thing.
            loss = (opts.alpha_mse * huber) + (opts.beta_stft * stft)
            # print(
            #     f"update run={capture_run} idx={idx} keras_model={opts.keras_run}"
            #     f" loss={loss} huber={huber} stft={stft}"
            # )
            db.set_losses(capture_run, idx, opts.keras_run, loss, huber, stft)
