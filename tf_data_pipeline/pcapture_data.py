from pathlib import Path
import zarr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from tqdm import tqdm

IN_D = 4
OUT_D = 1
IGNORE_FADE_LEN = 500


class ParametricCaptureData(object):

    def __init__(
        self,
        model_data_z: Path,
        seed: int = 123,
    ):
        try:
            self.model_data_z = zarr.open(str(model_data_z), mode="r")
        except zarr.errors.PathNotFoundError as e:
            print("files not found in", root_zarr_dir, str(e))
            raise e
        self.n_chunks = self.model_data_z.nchunks
        self.chunk_len = self.model_data_z.blocks[0].shape[0]
        # print("self.n_chunks", self.n_chunks, "self.chunk_len", self.chunk_len)
        self.rng = np.random.default_rng(seed=seed)

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
                # build x
                #  - triangle / core wave ( from capture )
                #  - cv value a_cv
                #  - cv_value b_cv
                #  - cv_value morph
                xs = data[:, 4]
                # build y
                #  - just morph output ( from capture )
                ys = data[:, 4:5]
                # yield
                yield xs, ys

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

    def tf_inference_dataset(self, batch_size: int = 1, cache_fname: str = None):
        """
        Generate all samples returned shape
        x - (1, SAMPLE_LEN, 4)
        y - (1, SAMPLE_LEN, 1)

        Args:
            seq_len: second axis for batch
            num_batches: total number of batches generated
            batch_size: batch size
        """

        def sample_generator():
            for c in range(self.n_chunks):
                data = self.model_data_z.blocks[c]
                # build x
                #  - triangle / core wave ( from capture )
                #  - cv value a_cv
                #  - cv_value b_cv
                #  - cv_value morph
                xs = data[:, :4]
                # build y
                #  - just morph output ( from capture )
                ys = data[:, 4:5]
                yield xs, ys

        ds = tf.data.Dataset.from_generator(
            sample_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.chunk_len, IN_D), dtype=tf.float32),
                tf.TensorSpec(shape=(self.chunk_len, OUT_D), dtype=tf.float32),
            ),
        )
        if cache_fname is not None:
            ds = ds.cache(cache_fname)
        if batch_size is not None:
            ds = ds.batch(batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--root-zarr-dir", type=str)
    parser.add_argument("--seq-len", type=int, default=5_120)
    parser.add_argument("--num-batches", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache", type=str)
    opts = parser.parse_args()
    print("opts", opts)

    pc_data = ParametricCaptureData(opts.root_zarr_dir)
    ds = pc_data.tf_training_dataset(
        seq_len=opts.seq_len,
        num_batches=opts.num_batches,
        batch_size=opts.batch_size,
        cache=opts.cache,
    )
    for _ in tqdm(ds, total=opts.num_batches):
        pass
