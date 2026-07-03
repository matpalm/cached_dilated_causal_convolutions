import numpy as np
import random
import datetime
from PIL import Image


def fade_in_out(buffer, fade_in_out_samples: int):
    num_samples, num_channels = buffer.shape

    # generate ( constant power ) quarter cycle cosine ramp
    t = np.linspace(0, np.pi, fade_in_out_samples, endpoint=False)
    fade_in_ramp = 0.5 - 0.5 * np.cos(t)
    fade_out_ramp = fade_in_ramp[::-1]

    # broadcast same ramp over all channels
    fade_in_ramp = fade_in_ramp[:, np.newaxis]
    fade_out_ramp = fade_out_ramp[:, np.newaxis]

    # apply
    buffer[:fade_in_out_samples] *= fade_in_ramp
    buffer[-fade_in_out_samples:] *= fade_out_ramp
    return buffer


def DTS():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def min_max_scale(a):
    raise Exception("use sklearn")
    diff = a.max() - a.min()
    return (a - a.min()) / diff
