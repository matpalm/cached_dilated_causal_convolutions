import argparse
from pathlib import Path
import zarr
import numpy as np

from pack_z_array import pack_z_array
from sample_db import SampleDB

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--srcs", type=str, required=True, nargs="+")
parser.add_argument("--dest", type=str, required=True)
opts = parser.parse_args()

db = SampleDB()


def combine(srcs, dest, dir_name_z, update_db: bool):

    # for src_z in srcs:
    #     if not src_z.exists():
    #         src_orig_d = Path(str(src_z).replace(".z", ""))
    #         if not src_orig_d.exists():
    #             raise Exception("src_orig_d", src_orig_d, "doesn't even exist")
    #         print("packing", src_orig_d, "to", src_z)
    #         pack_z_array(src_orig_d, src_z)

    axis = 0

    # open srcs; use first as assumed config for rest
    src_zarrs = [zarr.open(Path("runs") / f / dir_name_z, mode="r") for f in srcs]
    first_z = src_zarrs[0]
    base_shape = list(first_z.shape)
    chunks = first_z.chunks
    dtype = first_z.dtype

    # calc total len
    total_axis_length = sum(z.shape[axis] for z in src_zarrs)

    # calc target shape
    combined_shape = base_shape.copy()
    combined_shape[axis] = total_axis_length

    # open output
    dest_path = Path("runs") / dest / dir_name_z
    dest_path.mkdir(parents=True, exist_ok=True)
    dest_zarr = zarr.open(
        dest_path,
        mode="w",
        shape=tuple(combined_shape),
        chunks=chunks,
        dtype=dtype,
    )

    # stream data, aligned on chunks
    offset = 0
    db_idx_offset = 0
    for i, z in enumerate(src_zarrs):
        axis_length = z.shape[axis]
        target_slice = tuple(
            slice(offset, offset + axis_length) if idx == axis else slice(None)
            for idx in range(len(combined_shape))
        )
        dest_zarr[target_slice] = z
        if update_db:
            print("DB", srcs[i], "to", dest, "db_idx_offset", db_idx_offset)
            db.duplicate_run_with_idx_offset(
                src_run=srcs[i], dest_run=dest, idx_offset=db_idx_offset
            )
        offset += axis_length
        db_idx_offset += z.nchunks

    print("src_zarrs", [z.shape for z in src_zarrs])
    print("dest_zarr", dest_zarr.shape)


combine(opts.srcs, opts.dest, dir_name_z="cv_buffers.z", update_db=True)
combine(opts.srcs, opts.dest, dir_name_z="capture_buffers.z", update_db=False)
