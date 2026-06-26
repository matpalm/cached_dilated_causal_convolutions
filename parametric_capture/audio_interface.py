import sounddevice as sd
import numpy as np
from util import fade_in_out

class AudioInterface(object):

    NUM_IN_OUT_CHANNELS: int = 4
    BLOCKSIZE = 256  # try 256, 512, or 1024
    LATENCY_SEC = 0.04

    def __init__(self, sample_rate_hz=48_000):

        def scan_for_tiliqua():
            for i, device in enumerate(sd.query_devices()):
                print(i, device)
                if "tiliqua" in device["name"].lower():
                    return i
            raise Exception("not found")

        self.tiliqua_idx = scan_for_tiliqua()
        print("tiliqua_device", sd.query_devices(self.tiliqua_idx)["name"])

        self.sample_rate_hz = sample_rate_hz

    def send(self, buffer):
        buffer = fade_in_out(buffer, fade_num_samples=500)
        recorded_audio = sd.playrec(
            buffer.astype(np.float32),
            samplerate=self.sample_rate_hz,
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
