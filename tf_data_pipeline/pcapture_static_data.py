from pathlib import Path
import zarr
import numpy as np
import json
import tensorflow as tf
import pandas as pd

# from common.prio_replay import PrioExperienceReplay
from common.sample_db import SampleDB
from common.util import zarr_base_path_for
from .pcapture_data import model_data_block_to_xs_ys

IN_D = 4
OUT_D = 1
IGNORE_FADE_LEN = 500


class ParametricCaptureStaticData(object):

    def __init__(
        self,
        capture_run: str,
        keras_model: str,
        seed: int = 123,
    ):

        db = SampleDB()
        loss_rows = db.losses_for(capture_run, keras_model)
        self.losses = np.array([l.loss for l in loss_rows], dtype=np.float64)
        print("self.losses", self.losses)
        del db

        self.capture_run = capture_run
        self.model_data_z = zarr.open(
            zarr_base_path_for(capture_run) / "model_data_t.z", mode="r"
        )
        self.n_chunks = self.model_data_z.nchunks
        self.seq_len = self.model_data_z.blocks[0].shape[0]

        print(
            "capture_run",
            self.capture_run,
            "n_chunks",
            self.n_chunks,
            "chunk_len",
            self.seq_len,
            "|losses|",
            len(self.losses),
        )

        if len(self.losses) != self.n_chunks:
            raise Exception(
                "|losses| != n_chunks; either we have wrong losses or chunk_size of dest is wrong"
            )

        self.rng = np.random.default_rng(seed=seed)

        # compute static priorities
        high_loss_skew = 1.0  # 1 denotes skewing proportional to loss
        f64eps = np.finfo(np.float64).eps
        static_priorities = self.losses**high_loss_skew + f64eps
        print("static_priorities", static_priorities)

        # convert priorities to sampling probabilies ( just by normalisation )
        self.sampling_probabilies = static_priorities / static_priorities.sum()
        print("self.sampling_probabilies", self.sampling_probabilies)
        print("sum self.sampling_probabilies", self.sampling_probabilies.sum())
        # self.db = SampleDB()

        # since we are leaning heavily on converged ( ish ) loss of a large model
        # we can try to just calculate importance weights purely on that loss
        # i.e. regardless of where they came from; sobol, is_weights, uniform etc
        # this might be super naive... we'll see...
        bias_correction = 1.0  # no bias correction, we might need to drop this for stability? ( i.e more explore )
        num_examples = len(self.sampling_probabilies)
        unnormalised_static_importance_weights = (
            1.0 / num_examples * self.sampling_probabilies
        ) ** bias_correction
        print(
            "unnormalised_static_importance_weights",
            unnormalised_static_importance_weights,
        )
        print(
            "sum unnormalised_static_importance_weights",
            unnormalised_static_importance_weights.sum(),
        )
        self.static_importance_weights = (
            unnormalised_static_importance_weights
            / unnormalised_static_importance_weights.max()
        )
        print(
            "self.static_importance_weights",
            self.static_importance_weights,
        )
        print(
            "min",
            self.static_importance_weights.min(),
            "max",
            self.static_importance_weights.max(),
        )

        # read in debug mapping for src_runs ( which gives the src_run of each index )
        with open(zarr_base_path_for(capture_run) / "src_runs.json", "r") as f:
            src_runs = json.load(f)
        df = pd.DataFrame(
            zip(src_runs, self.sampling_probabilies, self.static_importance_weights),
            columns=["run", "sampling_probability", "static_importance_weight"],
        )
        df.to_csv("/tmp/weights.tsv", sep="\t", index=False)

        # size_next_po2 = 1 << (self.model_data_z.nchunks - 1).bit_length()
        # print(
        #     "size_next_po2", size_next_po2, "from n_chunks", self.model_data_z.nchunks
        # )
        # self.prio_replay = PrioExperienceReplay(
        #     size=size_next_po2  # , dump_log="/dev/shm/prio_replay_dump_log.tsv"
        # )
        # idxs, losses = [], []
        # for loss_row in self.db.losses_for(run=capture_run, model=keras_model):
        #     idxs.append(loss_row.idx)
        #     losses.append(loss_row.loss)
        # if len(idxs) != self.n_chunks:
        #     raise Exception(
        #         f"#scores in db={len(idxs)} doesn't match #chunks={self.n_chunks} for run {capture_run}"
        #     )
        # self.prio_replay.update(idxs, losses)

        # self.prio_replay.dump("init")

    def num_examples(self):
        return self.n_chunks

    def tf_training_dataset(self, seq_len: int, num_batches: int, batch_size: int):
        """
        Generate num_samples samples of shape (batch_size, seq_len, 4)
        sampling is done with importance sampling and

        Args:
            seq_len: second axis for batch
            num_batches: total number of batches generated
            batch_size: batch size
        """

        def sample_generator():
            for _ in range(num_batches):
                # note: since we are mixing with weights from sobol weights
                #       we don't max normalise to 1.0 until after
                idxs, weights = self.prio_replay.sample(batch_size, max_normalise=False)
                for idx, weight in zip(idxs, weights):
                    # sample offset/len
                    r_seq_from = self.rng.integers(
                        low=IGNORE_FADE_LEN,
                        high=self.chunk_len - IGNORE_FADE_LEN - seq_len,
                    )
                    r_seq_to = r_seq_from + seq_len
                    # grab relevant pieces
                    try:
                        block = self.model_data_z.blocks[idx]
                    except zarr.errors.BoundsCheckError as e:
                        print(
                            "self.capture_run",
                            self.capture_run,
                            "self.n_chunks",
                            self.n_chunks,
                            "idx",
                            idx,
                        )
                        raise e
                    data = block[r_seq_from:r_seq_to]
                    yield *model_data_block_to_xs_ys(data), idx, weight

        ds = tf.data.Dataset.from_generator(
            sample_generator,
            output_signature=(
                tf.TensorSpec(shape=(seq_len, IN_D), dtype=tf.float32),
                tf.TensorSpec(shape=(seq_len, OUT_D), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        )
        if num_batches is not None:
            ds = ds.batch(batch_size)

        # note: can't prefetch in importance sampling case since we need
        #       to explicitly update loss values

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
    pd = ParametricCaptureStaticData(capture_run="600", keras_model="231_keras/i9")
    # for x, y, idxs, weights in pd.tf_training_dataset(
    #     seq_len=100, num_batches=5, batch_size=8
    # ):
    #     print("idxs", idxs)
    #     print("weights", weights)
    #     print("x", x.shape, "y", y.shape)
