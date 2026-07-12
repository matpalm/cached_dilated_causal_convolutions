import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import os
import warnings
import json
from pathlib import Path
import tempfile

from tf_data_pipeline.quadrature_data import Embed2DQuadratureData, Waveform

print("tf", tf.__version__)

from qkeras_version.qkeras_model import create_dilated_model_from_config_and_latest_ckpt

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--tri-freq", type=float)
parser.add_argument(
    "--test-set",
    type=str,
    required=True,
    help="subdirectory under test_cvs/ containing cv_values.csv",
)
parser.add_argument("--run", type=Path)
parser.add_argument("--ckpt", type=Path)
parser.add_argument("--test-seq-len", type=int, default=400)
opts = parser.parse_args()
print("opts", opts)


def format_cv(v: float) -> str:
    # Use compact fixed precision that is filename-friendly.
    return f"{float(v):.4f}".rstrip("0").rstrip(".")


cv_csv_path = Path(__file__).parent / "test_cvs" / opts.test_set / "cv_values.csv"
if not cv_csv_path.exists():
    raise FileNotFoundError(f"cv_values.csv not found at {cv_csv_path}")

def load_cv_rows(csv_path: Path):
    df = pd.read_csv(csv_path)
    expected = ["a_cv", "b_cv", "morph_cv"]
    if all(c in df.columns for c in expected):
        rows = df[expected].to_numpy(dtype=np.float32)
    else:
        if df.shape[1] < 3:
            raise ValueError(
                "csv must have columns a_cv,b_cv,morph_cv or at least 3 columns"
            )
        rows = df.iloc[:, :3].to_numpy(dtype=np.float32)
    return [(float(a), float(b), float(m)) for a, b, m in rows]


cv_rows = load_cv_rows(cv_csv_path)
if len(cv_rows) == 0:
    raise ValueError(f"{cv_csv_path} has no rows")


img_output_path = Path("runs") / opts.run / "test_cvs" / opts.test_set
img_output_path.mkdir(parents=True, exist_ok=True)

for jpg in img_output_path.glob("*.jpg"):
    jpg.unlink()

test_model, receptive_field_size = create_dilated_model_from_config_and_latest_ckpt(
    opts.run
)

# the model needs to warm up so we have to run this many through
seq_len_plus_receptive_field = receptive_field_size + opts.test_seq_len

sample_rate = 48_000.0
amp = 0.53
n = np.arange(seq_len_plus_receptive_field, dtype=np.float32)
phase = np.mod(n * (opts.tri_freq / sample_rate), 1.0)
tri = amp * (2.0 * np.abs(2.0 * phase - 1.0) - 1.0)

x = np.empty((1, seq_len_plus_receptive_field, 4), dtype=np.float32)
x[0, :, 0] = tri
print("triangle[min,max]", float(tri.min()), float(tri.max()))
print("num cv rows", len(cv_rows))

for i, (a_cv, b_cv, morph_cv) in enumerate(cv_rows):
    x[0, :, 1] = np.float32(a_cv)
    x[0, :, 2] = np.float32(b_cv)
    x[0, :, 3] = np.float32(morph_cv)

    y_pred = test_model(x)
    y_pred = np.asarray(y_pred[0, :, 0], dtype=np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(n, tri, linewidth=1.5, label="input ch0")
    ax.plot(n, y_pred, linewidth=1.5, label="y_pred")
    ax.axvline(
        receptive_field_size,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"rf={receptive_field_size}",
    )
    ax.set_ylabel("value")
    ax.set_title(
        f"triangle={opts.tri_freq:.2f}Hz, a_cv={a_cv}, b_cv={b_cv}, morph_cv={morph_cv}"
    )
    ax.set_xlabel("sample index")
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    plot_path = (
        img_output_path
        / f"fixed_cv_{i:03d}_{format_cv(a_cv)}_{format_cv(b_cv)}_{format_cv(morph_cv)}.jpg"
    )
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"[{i+1}/{len(cv_rows)}] saved plot", plot_path)
