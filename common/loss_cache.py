from functools import cache
import numpy as np
import redis
import zarr
from scipy.signal import decimate
import tensorflow as tf

from common.util import model_data_z_path_for
from .losses import combined_masked_loss_terms


class CachedEdgeLoss(object):

    def __init__(self):
        self.redis_client = redis.Redis()
        self.cached_samples = CachedSamples(self.redis_client)
        loss_fns = combined_masked_loss_terms(
            receptive_field_size=None,
            use_huber_loss=True,
            alpha_mse=1.0,
            beta_stft=0.01,
            reduce_mean=True,
        )
        self.combined_loss_fn, self.huber_loss, self.stft_loss = loss_fns

    def get(self, r1, i1, r2, i2):
        r1, r2 = str(r1), str(r2)
        key1, key2 = f"{r1}:{i1}", f"{r2}:{i2}"
        if key1 > key2:
            return self.get(r2, i2, r1, i1)
        key = f"cel:{key1}:{key2}"
        value = self.redis_client.get(key)
        if value is None:
            value = self._get_miss(r1, i1, r2, i2)
            self.redis_client.set(key, value)

        combined_l, huber_l, stft_l = np.frombuffer(value, dtype=np.float32, count=3)
        return float(combined_l), float(huber_l), float(stft_l)

    def _get_miss(self, r1, i1, r2, i2):
        sample_1 = self.cached_samples.get(r1, i1)
        sample_2 = self.cached_samples.get(r2, i2)

        # convert to shape for losses; (1, S, 1)
        # and to f32 for stft
        sample_1 = tf.constant(sample_1[np.newaxis, :, np.newaxis].astype(np.float32))
        sample_2 = tf.constant(sample_2[np.newaxis, :, np.newaxis].astype(np.float32))

        return np.array(
            [
                self.combined_loss_fn(sample_1, sample_2),
                self.huber_loss(sample_1, sample_2),
                self.stft_loss(sample_1, sample_2),
            ],
            dtype=np.float32,
        ).tobytes()


class CachedSamples(object):

    def __init__(self, redis_client):
        self.redis_client = redis_client

    def get(self, run, idx):
        key = f"cs:{run}:{idx}"
        value = self.redis_client.get(key)
        if value is None:
            # print("CS MISS", run, idx)
            value = self._get_miss(run, idx)
            self.redis_client.set(key, value)
        # else:
        #     print("CS HIT", run, idx)
        return np.frombuffer(value, dtype=np.float16)

    @cache
    def chunk_buffer(self, run):
        return zarr.open(model_data_z_path_for(run), "r")

    def _get_miss(self, run, idx):
        buffer = self.chunk_buffer(run).blocks[idx]
        # just waveshaped
        buffer = buffer[:, 0]
        # remove fade in / out portion
        fade = 500
        buffer = buffer[fade:-fade]
        # decimate and convert to f16
        buffer = decimate(buffer, q=2)
        buffer = buffer.astype(np.float16)
        return buffer.tobytes()
