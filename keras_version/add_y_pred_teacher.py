import argparse
from pathlib import Path
import numpy as np
import zarr
import tensorflow as tf
import json
from tqdm import tqdm

from .keras_model import create_dilated_model_from_config_and_latest_ckpt
from common.util import zarr_base_path_for
from tf_data_pipeline.pcapture_data import model_data_block_to_xs_ys
from common.sample_db import SampleDB

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", type=str, required=True)
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

db = SampleDB()

# build inference model and restore ckpt
inference_model = create_dilated_model_from_config_and_latest_ckpt(opts.model)

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
seq_len, num_fields = src_zarrs[0].blocks[0].shape
print("seq_len", seq_len)
print("num_fields", num_fields)
dtype = src_zarrs[0].dtype
total_src_samples = sum(z.shape[0] for z in src_zarrs)  # total rows across srcs
total_src_chunks = sum(z.nchunks for z in src_zarrs)  # one block == one sample

# target shape is the combined number of samples, but with extra feature column
combined_shape = (total_src_samples, num_fields + 1)
print("combined_shape", combined_shape)

# chunk size for output is same as input, but with extra feature column
combined_chunks = (seq_len, num_fields + 1)
print("combined_chunks", combined_chunks)

# open output
dest_path = Path(
    zarr_base_path_for(opts.dest_run, check_exists=False) / "model_data_t.z"
)
dest_path.mkdir(parents=True, exist_ok=True)
dest_zarr = zarr.open(
    dest_path,
    mode="w",
    shape=combined_shape,
    chunks=combined_chunks,
    dtype=dtype,
)

# duplicate rows in db
write_idx = 0
for src_zarr, run_str in zip(src_zarrs, opts.src_runs):
    print("duplicate_run_with_idx_offset", run_str, write_idx)
    db.duplicate_run_with_idx_offset(
        src_run=run_str, dest_run=opts.dest_run, idx_offset=write_idx
    )
    write_idx += src_zarr.nchunks

# for debugging later we make a list [src_run, src_run, ...]
# so we can map the rows in dest_run back to src_run for debugging
src_runs = []
write_idx = 0
for src_model_data_z, run_str in zip(src_zarrs, opts.src_runs):
    n_blocks = src_model_data_z.nchunks
    batch_size = 32
    starts = list(range(0, n_blocks, batch_size))
    for start in tqdm(starts, desc=f"processing {run_str}"):
        end = min(start + batch_size, n_blocks)
        eg_b = src_model_data_z.blocks[start:end].reshape((-1, seq_len, num_fields))
        x_b, _y_true_b = model_data_block_to_xs_ys(eg_b)
        y_teacher_b = inference_model(x_b, training=False).numpy()
        batch_size = len(x_b)  # may be <full for last batch in runs
        out_b = np.concatenate([eg_b, y_teacher_b], axis=-1)
        for b in range(batch_size):
            dest_zarr.blocks[write_idx] = out_b[b].reshape((-1, num_fields + 1))
            write_idx += 1
            src_runs.append(str(run_str))
        # print(
        #     "<process_batch start",
        #     start,
        #     "batch_size",
        #     batch_size,
        #     "write_idx",
        #     write_idx,
        # )

# TODO: src_runs.json is a bad idea, much better to have a lookup table in db
#       we could set for anywhere we combine runs
with open(zarr_base_path_for(opts.dest_run) / "src_runs.json", "w") as f:
    json.dump(src_runs, fp=f)

print("wrote", write_idx, "samples to", dest_path)
assert write_idx == total_src_chunks, (write_idx, total_src_chunks)
