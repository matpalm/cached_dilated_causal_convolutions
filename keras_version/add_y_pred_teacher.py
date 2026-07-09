import argparse
from pathlib import Path
import numpy as np
import zarr
import tensorflow as tf

from .keras_model import create_dilated_model_from_config_and_latest_ckpt
from common.util import zarr_base_path_for
from tf_data_pipeline.pcapture_data import model_data_block_to_xs_ys
from common.sample_db import SampleDB

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--keras-run", type=str, required=True)
parser.add_argument(
    "--src-runs",
    type=Path,
    required=True,
    nargs="+",
    help="source for model_data.z srcs",
)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument(
    "--dest-run",
    type=Path,
    required=True,
    help="destination zarr with model_data_t.z with y_pred_teacher",
)
parser.add_argument(
    "--input-epsilon",
    type=float,
    default=0.05,
    help="for each example material y_pred from x we also calculate x+e to give student a finite-difference like signal",
)
opts = parser.parse_args()
print("opts", opts)


# build inference model and restore ckpt
inference_model = create_dilated_model_from_config_and_latest_ckpt(opts.keras_run)

# materialise (x, [y_true, y_teacher]) for every chunk across all capture runs


# def add_teacher_target(x_b, y_true_b):
#     """
#     Emit TWO examples per input chunk: the base (x, [y_true, teacher(x)]) and a
#     perturbed (x+e, [y_true, teacher(x+e)]). Both are stacked along the batch
#     axis, so the downstream .unbatch() yields them as separate elements.

#     Args:
#         x_b: (B, chunk_len, IN_D)
#         y_true_b: (B, chunk_len, 1)
#     """
#     # run teacher for main x_b
#     y_teacher_b = inference_model(x_b, training=False)

#     # perturb the core wave (channel 0) by +epsilon and rerun
#     x_e_b = x_b + tf.constant(
#         [opts.input_epsilon] + [0.0] * (IN_D - 1), dtype=x_b.dtype
#     )
#     y_teacher_e_b = inference_model(x_e_b, training=False)

#     # stack base + perturbed along the batch axis
#     # can unbatch them straight after
#     x_out = tf.concat([x_b, x_b_e], axis=0)
#     y_out = tf.concat([y_b, y_b_e], axis=0)
#     return x_out, y_out


# open srcs; use first as assumed config for rest
src_zarrs = [zarr.open(zarr_base_path_for(s) / "model_data.z") for s in opts.src_runs]
first_z = src_zarrs[0]
base_shape = first_z.shape
print("base_zarr_shape", base_shape)
chunks = first_z.chunks
sample_shape = first_z.blocks[0].shape
print("sample_shape", sample_shape)
dtype = first_z.dtype
num_fields = base_shape[-1]
total_samples = sum(z.shape[0] for z in src_zarrs)  # total rows across srcs
total_blocks = sum(z.nchunks for z in src_zarrs)  # one block == one sample

# calc target shape
combined_shape = list(base_shape)
combined_shape[0] = total_samples
combined_shape[-1] = num_fields + 1
print("combined_shape", combined_shape)

# open output
dest_path = Path(
    zarr_base_path_for(opts.dest_run, check_exists=False) / "model_data_t.z"
)
dest_path.mkdir(parents=True, exist_ok=True)
dest_zarr = zarr.open(
    dest_path,
    mode="w",
    shape=tuple(combined_shape),
    chunks=chunks,
    dtype=dtype,
)


def all_src_examples_gen(batch_size: int = 8):
    for src_model_data_z, run_str in zip(src_zarrs, opts.src_runs):
        print("processing", run_str)
        n_blocks = src_model_data_z.nchunks
        for start in range(0, n_blocks, batch_size):
            end = min(start + batch_size, n_blocks)
            yield src_model_data_z.blocks[start:end].reshape((-1, *sample_shape))


write_idx = 0
for eg_b in all_src_examples_gen(batch_size=opts.batch_size):
    x_b, y_true_b = model_data_block_to_xs_ys(eg_b)
    y_teacher_b = inference_model(x_b, training=False).numpy()
    batch_size = len(x_b)  # may be <full for last batch in runs
    out_b = np.concatenate([eg_b, y_teacher_b], axis=-1)
    for b in range(batch_size):
        dest_zarr.blocks[write_idx] = out_b[b].reshape((-1, 6))
        write_idx += 1
print("wrote", write_idx, "samples to", dest_path)

# also write losses numpy; one for each samples
db = SampleDB()
losses = []
for s in opts.src_runs:
    losses.extend([l.loss for l in db.losses_for(s, model=opts.keras_run)])
losses = np.array(losses)
total_losses = len(losses)
np_path = zarr_base_path_for(opts.dest_run) / "model_data_t.losses.npy"
np.save(np_path, losses)
print(f"wrote losses ({losses.shape}) to {np_path}")

assert write_idx == total_blocks == total_losses, (
    write_idx,
    total_blocks,
    total_losses,
)
