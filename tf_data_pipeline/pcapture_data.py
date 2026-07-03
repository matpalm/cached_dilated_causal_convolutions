from pathlib import Path
import zarr
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import List

from common.util import model_data_z_path_for

IN_D = 4
OUT_D = 1
IGNORE_FADE_LEN = 500


def model_data_block_to_xs_ys(data):
    # build x
    #  - triangle / core wave ( from capture )
    #  - cv value a_cv
    #  - cv_value b_cv
    #  - cv_value morph
    xs = data[:, :4]
    # build y
    #  - just morph output ( from capture )
    ys = data[:, 4:5]
    return xs, ys


# too ineffecient to use this?
# def interleaved_datasets(tf_datasets, repeats: List[bool], weights: List[int]):
#     """sample from the N ParametricCaptureData(model_data_z) proportional to weights"""
#     arg_lens = [len(tf_datasets), len(repeats), len(weights)]
#     if len(set(arg_lens)) != 1:
#         raise Exception("all lens must match")
#     # set repeat on requested datasets
#     # be careful! if they are all True this dataset is infinite
#     tf_datasets = [ds.repeat() if r else ds for ds, r in zip(tf_datasets, repeats)]
#     probabilities = [w / sum(weights) for w in weights]
#     return tf.data.Dataset.sample_from_datasets(
#         datasets=tf_datasets,
#         weights=probabilities,
#         seed=385,
#         stop_on_empty_dataset=True,
#     )


class ParametricCaptureData(object):

    def __init__(
        self,
        capture_run: str,
        seed: int = 123,
    ):
        self.capture_run = capture_run
        try:
            self.model_data_z = zarr.open(model_data_z_path_for(capture_run), mode="r")
        except zarr.errors.PathNotFoundError as e:
            print("files not found in", str(model_data_z), str(e))
            raise e
        self.n_chunks = self.model_data_z.nchunks
        self.chunk_len = self.model_data_z.blocks[0].shape[0]
        # print("self.n_chunks", self.n_chunks, "self.chunk_len", self.chunk_len)
        self.rng = np.random.default_rng(seed=seed)

    def num_examples(self):
        return self.n_chunks

    def tf_training_dataset(
        self, seq_len: int, num_batches: int, batch_size: int, cache_fname: str = None
    ):
        """
        Generate num_samples samples of shape (batch_size, seq_len, 4)

        Args:
            seq_len: second axis for batch
            num_batches: total number of batches generated
            batch_size: batch size
        """

        def sample_generator():
            for _ in range(num_batches * batch_size):
                # sample chunk and sample offset/len
                r_chunk = self.rng.integers(low=0, high=self.n_chunks)
                r_seq_from = self.rng.integers(
                    low=IGNORE_FADE_LEN, high=self.chunk_len - IGNORE_FADE_LEN - seq_len
                )
                r_seq_to = r_seq_from + seq_len
                # grab relevant pieces
                data = self.model_data_z.blocks[r_chunk][r_seq_from:r_seq_to]
                yield model_data_block_to_xs_ys(data)

        ds = tf.data.Dataset.from_generator(
            sample_generator,
            output_signature=(
                tf.TensorSpec(shape=(seq_len, IN_D), dtype=tf.float32),
                tf.TensorSpec(shape=(seq_len, OUT_D), dtype=tf.float32),
            ),
        )
        if cache_fname is not None:
            ds = ds.cache(cache_fname)
        if num_batches is not None:
            ds = ds.batch(batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def tf_inference_dataset(
        self,
        batch_size: int = 1,
        cache_fname: str = None,
        return_sample_info: bool = False,
    ):
        """
        Generate all samples, once, with full returned shape
        x - (1, SAMPLE_LEN, 4)
        y - (1, SAMPLE_LEN, 1)

        Args:
            seq_len: second axis for batch
            num_batches: total number of batches generated
            batch_size: batch size
            return_sample_info: if True return (x, y, model_data_z, idx) otherwise return normal (x, y)
        """

        def sample_generator():
            for c in range(self.n_chunks):
                data = self.model_data_z.blocks[c]
                if return_sample_info:
                    yield *model_data_block_to_xs_ys(data), self.capture_run, c
                else:
                    yield model_data_block_to_xs_ys(data)

        output_signature = [
            tf.TensorSpec(shape=(self.chunk_len, IN_D), dtype=tf.float32),
            tf.TensorSpec(shape=(self.chunk_len, OUT_D), dtype=tf.float32),
        ]
        if return_sample_info:
            output_signature.append(tf.TensorSpec(shape=(), dtype=tf.string))
            output_signature.append(tf.TensorSpec(shape=(), dtype=tf.int32))

        ds = tf.data.Dataset.from_generator(
            sample_generator, output_signature=tuple(output_signature)
        )

        if cache_fname is not None:
            ds = ds.cache(cache_fname)
        if batch_size is not None:
            ds = ds.batch(batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


if __name__ == "__main__":
    import argparse

    def is_true_str(value):
        return value.lower() == "true"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--root-zarr-dir", type=Path, nargs="+")
    parser.add_argument("--repeats", type=is_true_str, nargs="+")
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--batch-size", type=int, default=4)
    opts = parser.parse_args()
    print("opts", opts)

    assert len(opts.root_zarr_dir) == len(opts.repeats)
    assert len(opts.repeats) == len(opts.weights)

    datasets = []
    for zarr_d in opts.root_zarr_dir:
        pc_data = ParametricCaptureData(zarr_d)
        datasets.append(
            pc_data.tf_inference_dataset(batch_size=None, return_sample_info=True)
        )

    ds = interleaved_datasets(datasets, opts.repeats, opts.weights).batch(4)

    for i, r in enumerate(ds):
        if len(r) == 2:
            x, y = r
            print(i, "x", x.shape, "y", y.shape)
        elif len(r) == 4:
            x, y, m, c = r
            print(i, "x", x.shape, "y", y.shape, "m", m, "c", c)
