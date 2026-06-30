import argparse
import json
import zarr
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import seaborn as sns
import warnings

from .keras_model import create_dilated_model
from tf_data_pipeline.pcapture_data import model_data_block_to_xs_ys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--model-ckpt", type=Path, required=True)
parser.add_argument("--model-data-z", type=Path, required=True)
parser.add_argument("--sample-id", type=int, required=True)
parser.add_argument("--plot-offset", type=int, default=20_000)
parser.add_argument("--plot-len", type=int, default=2_000)
parser.add_argument("--plot-fname", type=str, required=True)
opts = parser.parse_args()
print("opts", opts)

with open("runs" / opts.run / "model_config.json", "r") as f:
    model_config = json.load(f)

inference_model = create_dilated_model(**model_config)
inference_model.load_weights(str(opts.model_ckpt))

model_data_z = zarr.open(str(opts.model_data_z), mode="r")
try:
    sample = model_data_z.blocks[opts.sample_id]
except IndexError as e:
    print("model_data_z.nchunks", model_data_z.nchunks)
    raise e
print("sample", sample.shape)
x, y_true = model_data_block_to_xs_ys(sample)
x = np.expand_dims(x, 0)
y_true = np.expand_dims(y_true, 0)
print("x", x.shape, "y_true", y_true.shape)
y_pred = inference_model(x)
print("y_pred", y_pred.shape)

df = pd.DataFrame()
r1 = opts.plot_offset
r2 = opts.plot_offset + opts.plot_len
df["y_true"] = y_true[0, r1:r2, 0]
df["y_pred"] = y_pred[0, r1:r2, 0]
df["n"] = range(opts.plot_len)
wide_df = pd.melt(
    df,
    id_vars=["n"],
    #            value_vars=["tri", "y_pred", "y_true", "a_cv", "b_cv", "morph"],
    value_vars=["y_pred", "y_true"],
)
with io.BytesIO() as img_buffer:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        fig, ax = plt.subplots(figsize=(30, 5))
        sns.lineplot(wide_df, x="n", y="value", hue="variable", ax=ax)
        ax.set_ylim((-1.1, 1.1))
        fig.savefig(img_buffer, format="png")
        plt.close(fig)
    img_buffer.seek(0)
    pil_img = Image.open(img_buffer).convert("RGB")
pil_img.save(opts.plot_fname)
