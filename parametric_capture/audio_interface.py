import sounddevice as sd
import numpy as np

from .util import fade_in_out

SAMPLE_RATE_HZ = 48_000

class AudioInterface(object):

    NUM_IN_OUT_CHANNELS: int = 4
    BLOCKSIZE = 256  # try 256, 512, or 1024
    LATENCY_SEC = 0.04

    def __init__(self):

        def scan_for_tiliqua():
            for i, device in enumerate(sd.query_devices()):
                if "tiliqua" in device["name"].lower():
                    return i
            raise Exception("not found")

        self.tiliqua_idx = scan_for_tiliqua()
        print("tiliqua_device", sd.query_devices(self.tiliqua_idx)["name"])

    def send(self, buffer, fade_in_out_samples=500):
        buffer = fade_in_out(buffer, fade_in_out_samples=500)
        recorded_audio = sd.playrec(
            buffer.astype(np.float32),
            samplerate=SAMPLE_RATE_HZ,
            channels=self.NUM_IN_OUT_CHANNELS,
            dtype="float32",
            device=(self.tiliqua_idx, self.tiliqua_idx),
            blocksize=self.BLOCKSIZE,
            latency=(self.LATENCY_SEC, self.LATENCY_SEC),
        )
        sd.wait()
        return recorded_audio

    def num_channels(self):
        return self.NUM_IN_OUT_CHANNELS
