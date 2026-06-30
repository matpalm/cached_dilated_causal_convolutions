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
from qkeras_version.losses import masked_multires_stft_loss

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--model-ckpt", type=Path, required=True)
parser.add_argument("--model-data-z", type=Path, required=True)
parser.add_argument("--losses-tsv", type=Path, required=True)
opts = parser.parse_args()
print("opts", opts)

with open("runs" / opts.run / "model_config.json", "r") as f:
    model_config = json.load(f)

inference_model = create_dilated_model(**model_config)
inference_model.load_weights(str(opts.model_ckpt))

print(inference_model.summary())

tsv_out = open(opts.losses_tsv, "w")
print("mse\thuber\tstft", file=tsv_out)

# delta=0.1 reasonable for (-1, .1)
huber_loss = Huber(delta=0.1)
stft_loss = masked_multires_stft_loss()
pcapture_data = ParametricCaptureData(model_data_z=opts.model_data_z)
batch_size = 32
ds = pcapture_data.tf_inference_dataset(batch_size=batch_size)
num_batches = math.ceil(pcapture_data.n_chunks / batch_size)
for x_batch, y_true_batch in tqdm(ds, total=num_batches):
    y_pred_batch = inference_model(x_batch)
    batch_n = y_true_batch.shape[0]
    for i in range(batch_n):
        y_true = y_true_batch[i : i + 1]
        y_pred = y_pred_batch[i : i + 1]
        mean_mse = np.mean(MSE(y_true, y_pred))
        mean_huber = huber_loss(y_true, y_pred)
        stft = stft_loss(y_true, y_pred)
        losses_f = map(float, [mean_mse, mean_huber, stft])
        losses_s = map(str, losses_f)
        print("\t".join(losses_s), file=tsv_out)
tsv_out.close()
