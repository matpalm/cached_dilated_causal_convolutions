from pathlib import Path
import zarr
import numpy as np
import json
import tensorflow as tf
import pandas as pd

from common.sample_db import SampleDB
from common.util import zarr_base_path_for

IN_D = 4
OUT_D = 1  # TODO: change to 2 for y_teacher_pred
IGNORE_FADE_LEN = 500

# generate samples based on materialised sampling probabilities / importance sampling
# weights based on converged model loss. dataset includes y_teacher as possible output


def model_data_block_to_xs_ys(data, emit_y_teacher_pred: bool):
    """
    Args:
        data: chunk from zarr model_data_t.z
        emit_y_teacher_pred: if true emit y_teacher_pred, else emit y_true
    """
    # build x
    #  - triangle / core wave ( from capture )
    #  - cv value a_cv
    #  - cv_value b_cv
    #  - cv_value morph
    xs = data[..., :4]
    # build y
    #  - y_teacher_pred morph output ( from capture ) for NOW or
    #  - y_true morph output ( from capture )
    if emit_y_teacher_pred:
        ys = data[..., 5:6]
    else:
        ys = data[..., 4:5]
    return xs, ys


class ParametricCaptureStaticData(object):

    def __init__(
        self,
        capture_run: str,
        keras_model: str,
        seed: int = 123,
    ):

        # TOOD: given the static data includes a y_teacher baked in it only
        #       makes sense to use one model. so these losses should be
        #       be written in add_y_pred and then read from capture_run model_data_t (?)

        db = SampleDB()
        loss_rows = db.losses_for(capture_run, keras_model)
        self.losses = np.array([l.loss for l in loss_rows], dtype=np.float64)
        print("self.losses", self.losses)
        if len(self.losses) == 0:
            raise Exception(
                f"no scores in db for run={capture_run} model={keras_model} ?"
            )
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
        # TODO: try high_loss_skew in 0.4, 0.7 range
        #  0.0 => uniform ( ignore loss )
        #  1.0 => denotes skewing proportional to loss
        high_loss_skew = 1.0
        f64eps = np.finfo(np.float64).eps
        static_priorities = self.losses**high_loss_skew + f64eps

        # convert priorities to sampling probabilies ( just by normalisation )
        self.sampling_probabilies = static_priorities / static_priorities.sum()

        # since we are leaning heavily on converged ( ish ) loss of a large model
        # we can try to just calculate importance weights purely on that loss
        # i.e. regardless of where they came from; sobol, is_weights, uniform etc
        # this might be super naive... we'll see...
        # TODO: try bias_correction in 0.5, 1.0
        #  0 => w_i=1 for all => keeps all bias from sampling prio
        #  1 => full correction => weighting cancels out sampling prio
        bias_correction = 1.0
        num_examples = len(self.sampling_probabilies)
        unnormalised_static_importance_weights = (
            1.0 / (num_examples * self.sampling_probabilies)
        ) ** bias_correction
        self.static_importance_weights = (
            unnormalised_static_importance_weights
            / unnormalised_static_importance_weights.max()
        )

        # read in debug mapping for src_runs ( which gives the src_run of each index )
        with open(zarr_base_path_for(capture_run) / "src_runs.json", "r") as f:
            src_runs = json.load(f)
        # write key arrays for debugging
        df = pd.DataFrame(
            zip(src_runs, self.sampling_probabilies, self.static_importance_weights),
            columns=["run", "sampling_probability", "static_importance_weight"],
        )
        df.to_csv("/tmp/weights.tsv", sep="\t", index=False)

        if (
            self.n_chunks
            != len(self.sampling_probabilies)
            != len(self.static_importance_weights)
        ):
            raise Exception(
                "mismatch between n_chunks, sampling_probabilies, static_importance_weights"
            )

    def num_examples(self):
        return self.n_chunks

    def tf_training_dataset(
        self,
        seq_len: int,
        num_batches: int,
        batch_size: int,
        emit_weights: bool,
        emit_y_teacher_pred: bool,
    ):
        """
        Generate num_samples samples of shape (batch_size, seq_len, 4)
        sampling is done with statically derived importance sampling probabilities
        and importance weights.

        Args:
            seq_len: second axis for batch
            num_batches: total number of batches generated
            batch_size: batch size
            emit_weight: if true return _weight as 3rd
            emit_y_teacher_pred: if y_true emit
        """

        def sample_generator():
            for _ in range(num_batches * batch_size):
                # sample idx
                idx = self.rng.choice(self.n_chunks, p=self.sampling_probabilies)
                # sample offset/len
                r_seq_from = self.rng.integers(
                    low=IGNORE_FADE_LEN,
                    high=self.seq_len - IGNORE_FADE_LEN - seq_len,
                )
                r_seq_to = r_seq_from + seq_len
                # grab relevant pieces
                try:
                    block = self.model_data_z.blocks[idx]
                    data = block[r_seq_from:r_seq_to]
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
                # return with weight for training and either y_true or y_teacher_pred
                xs_ys = model_data_block_to_xs_ys(data, emit_y_teacher_pred)
                if emit_weights:
                    weight = self.static_importance_weights[idx]
                    yield *xs_ys, weight
                else:
                    yield xs_ys

        output_signature = [
            tf.TensorSpec(shape=(seq_len, IN_D), dtype=tf.float16),
            tf.TensorSpec(shape=(seq_len, OUT_D), dtype=tf.float16),
        ]
        if emit_weights:
            output_signature.append(tf.TensorSpec(shape=(), dtype=tf.float32))

        ds = tf.data.Dataset.from_generator(
            sample_generator, output_signature=tuple(output_signature)
        )

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
            return_sample_info: if True return (x, y, model_data_z, idx, static_weight) otherwise return normal (x, y)
        """

        def sample_generator():
            for c in range(self.n_chunks):
                data = self.model_data_z.blocks[c]
                weight = self.static_importance_weights[idx]
                if return_sample_info:
                    yield *model_data_block_to_xs_ys(data), self.capture_run, c, weight
                else:
                    yield model_data_block_to_xs_ys(data)

        output_signature = [
            tf.TensorSpec(shape=(self.seq_len, IN_D), dtype=tf.float32),
            tf.TensorSpec(shape=(self.seq_len, OUT_D), dtype=tf.float32),
        ]
        if return_sample_info:
            output_signature.append(tf.TensorSpec(shape=(), dtype=tf.string))
            output_signature.append(tf.TensorSpec(shape=(), dtype=tf.int32))
            output_signature.append(tf.TensorSpec(shape=(), dtype=tf.float32))

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

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    opts = parser.parse_args()
    print("opts", opts)
    pd = ParametricCaptureStaticData(capture_run=opts.run, keras_model=opts.model)

    ds = pd.tf_training_dataset(seq_len=64, num_batches=4, batch_size=4)
    for xs, ys, weights in ds:
        print(xs.shape, ys.shape, weights)

    # for x, y, idxs, weights in pd.tf_training_dataset(
    #     seq_len=100, num_batches=5, batch_size=8
    # ):
    #     print("idxs", idxs)
    #     print("weights", weights)
    #     print("x", x.shape, "y", y.shape)
