import argparse
import json
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
parser.add_argument("--model-data-z", type=Path, required=True)
parser.add_argument("--losses-tsv", type=Path, required=True)
opts = parser.parse_args()
print("opts", opts)

with open("runs" / opts.run / "model_config.json", "r") as f:
    model_config = json.load(f)

inference_model = create_dilated_model(**model_config)
print(inference_model.summary())

# delta=0.1 reasonable for (-1, .1)
huber_loss = Huber(delta=0.1)
stft_loss = masked_multires_stft_loss()
pcapture_data = ParametricCaptureData(model_data_z=opts.model_data_z)
records = []
for x, y_true in tqdm(
    pcapture_data.tf_inference_dataset(), total=pcapture_data.n_chunks
):
    # TODO: support batching
    y_pred = inference_model(x)
    # print("y_true", y_true.shape, y_true[0][:10])
    # print("y_pred", y_pred.shape, y_pred[0][:10])
    mean_mse = np.mean(MSE(y_true, y_pred))
    # print("mean_mse", float(mean_mse))
    mean_huber = huber_loss(y_true, y_pred)
    # print("mean_huber", float(mean_huber))
    stft = stft_loss(y_true, y_pred)
    # print("stft_loss", float(stft_loss))
    records.append(
        {"mse": float(mean_mse), "huber": float(mean_huber), "sftf": float(stft)}
    )

pd.DataFrame(records).to_csv(opts.losses_tsv, sep="\t", index=False)
