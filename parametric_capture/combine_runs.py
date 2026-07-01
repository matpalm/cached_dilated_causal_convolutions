import argparse
from pathlib import Path
import zarr
import numpy as np
from pack_z_array import pack_z_array

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--srcs", type=str, required=True, nargs="+")
parser.add_argument("--dest", type=str, required=True)
opts = parser.parse_args()

raise Exception("add support in sample_db")

srcs = [Path("runs") / s for s in opts.srcs]

dest = Path("runs") / opts.dest
dest.mkdir(parents=True, exist_ok=True)

def combine(srcs, dest):

    for src_z in srcs:
        if not src_z.exists():
            src_orig_d = Path(str(src_z).replace(".z", ""))
            if not src_orig_d.exists():
                raise Exception("src_orig_d", src_orig_d, "doesn't even exist")
            print("packing", src_orig_d, "to", src_z)
            pack_z_array(src_orig_d, src_z)

    axis = 0

    # open srcs; use first as assumed config for rest
    src_zarrs = [zarr.open(f, mode="r") for f in srcs]
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
    dest_zarr = zarr.open(
        dest, mode="w", shape=tuple(combined_shape), chunks=chunks, dtype=dtype
    )

    # stream data, aligned on chunks
    offset = 0
    for i, z in enumerate(src_zarrs):
        axis_length = z.shape[axis]
        target_slice = tuple(
            slice(offset, offset + axis_length) if idx == axis else slice(None)
            for idx in range(len(combined_shape))
        )
        dest_zarr[target_slice] = z
        offset += axis_length

    print("src_zarrs", [z.shape for z in src_zarrs])
    print("dest_zarr", dest_zarr.shape)


combine(srcs=[r / "cv_buffers.z" for r in srcs], dest=dest / "cv_buffers.z")
combine(
    srcs=[r / "capture_buffers.z" for r in srcs],
    dest=dest / "capture_buffers.z",
)
