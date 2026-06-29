from pathlib import Path
import zarr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import lfilter

IN_D = 4
OUT_D = 1
IGNORE_FADE_LEN = 500


class ParametricCaptureData(object):

    def __init__(
        self,
        root_zarr_dir: Path,
        seed: int = 123,
    ):
        root_zarr_dir = Path(root_zarr_dir)
        try:
            self.model_data_z = zarr.open(root_zarr_dir / "model_data.z", mode="r")
        except zarr.errors.PathNotFoundError as e:
            print("files not found in", root_zarr_dir)
            raise e
        self.n_chunks = self.model_data_z.nchunks
        self.chunk_len = self.model_data_z.blocks[0].shape[0]
        print("self.n_chunks", self.n_chunks, "self.chunk_len", self.chunk_len)
        self.rng = np.random.default_rng(seed=seed)

    def tf_dataset(self, seq_len: int, num_batches: int, batch_size: int):
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
                xs = np.empty((seq_len, IN_D), dtype=np.float32)
                for c in range(4):
                    xs[:, c] = data[:, c]
                # build y
                #  - just morph output ( from capture )
                ys = np.empty((seq_len, OUT_D), dtype=np.float32)
                ys[:, 0] = data[:, 4]
                # yield
                yield xs, ys

        return sample_generator()

        # ds = tf.data.Dataset.from_generator(
        #     sample_generator,
        #     output_signature=(
        #         tf.TensorSpec(shape=(seq_len, IN_D), dtype=tf.float32),
        #         tf.TensorSpec(shape=(seq_len, OUT_D), dtype=tf.float32),
        #     ),
        # )
        # ds = ds.batch(batch_size)
        # return ds.prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    pc_data = ParametricCaptureData(
        root_zarr_dir="/home/mat/dev/cached_dilated_causal_convolutions/parametric_capture/runs/001"
    )

    ds = pc_data.tf_dataset(seq_len=1_000, num_batches=16, batch_size=1)
    for n, (xs, ys) in enumerate(ds):

        combined = np.concatenate([xs, ys], axis=1)  # (1000, 5)
        labels = ["tri", "a_cv", "b_cv", "morph_cv", "morph_out"]
        steps = np.arange(combined.shape[0])

        plt.figure(figsize=(12, 6))
        for i, label in enumerate(labels):
            plt.plot(steps, combined[:, i], label=label, linewidth=1.2)

        plt.xlim(0, 999)
        plt.ylim(-1.0, 1.0)
        plt.xlabel("step")
        plt.ylabel("value")
        plt.title("Parametric Capture: xs + ys")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"pcapture_xs_ys_plot.{n:02d}.jpg", dpi=300, format="jpg")
        plt.close()
