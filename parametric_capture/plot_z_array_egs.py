import os
from pathlib import Path
import zarr
import numpy as np
import argparse
from functools import cache

from .plotting import plot, collage, plot_spectrogram
from common.util import zarr_base_path_for, zarr_buffer_fields

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--egs", type=str, nargs="+", required=True, help="run_idx to plot")
parser.add_argument("--plot-offset", type=int, default=10_000)
parser.add_argument("--plot-len", type=int, default=5_000)

opts = parser.parse_args()


@cache
def open_zarr(run, z_name):
    try:
        return zarr.open(str(zarr_base_path_for(run) / z_name), "r")
    except zarr.errors.PathNotFoundError:
        print("NOT FOUND", run, z_name)
        return None


pil_plots = []

for eg in opts.egs:
    run, idx = eg.split("_")
    idx = int(idx)

    plots = []
    for z_name in [
        "cv_buffers.z",
        "capture_buffers.z",
        "model_data.z",
        "model_data_t.z",
    ]:
        z = open_zarr(run, z_name)
        ch_names = zarr_buffer_fields(z_name)._fields
        if z:
            plots.append(
                plot(
                    z.blocks[idx],
                    title=z_name,
                    ch_names=ch_names,
                    plot_offset=opts.plot_offset,
                    plot_len=opts.plot_len,
                )
            )
    collage_fname = f"zarr_eg.buffer.r{run}_i{idx}.jpg"
    collage(plots, side_by_side=True).save(collage_fname)
    print("plotted", collage_fname)

    # plots = [
    #     plot_spectrogram(
    #         cv_z.blocks[idx][:, 0],
    #         title="cv_buffers",
    #         plot_offset=opts.plot_offset,
    #         plot_len=opts.plot_len,
    #     ),
    #     plot_spectrogram(
    #         cb_z.blocks[idx][:, 0],
    #         title="capture_buffers",
    #         plot_offset=opts.plot_offset,
    #         plot_len=opts.plot_len,
    #     ),
    #     plot_spectrogram(
    #         md_z.blocks[idx][:, 0],
    #         title="model_data",
    #         plot_offset=opts.plot_offset,
    #         plot_len=opts.plot_len,
    #     ),
    # ]
    # collage_fname = f"zarr_eg.spectrogram.r{run}_i{idx}.jpg"
    # collage(plots, stacked=True).save(collage_fname)
    # print("plotted", collage_fname)
