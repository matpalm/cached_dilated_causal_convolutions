import random
import argparse
import sys

from audio_interface import AudioInterface
from sampling import SobolSampler
from plotting import *
from util import *
from sample_db import SampleDB

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=str, required=True)
opts = parser.parse_args()

cv_samples = []
for line in sys.stdin.readlines():
    cv_samples.append(list(map(float, line.split(","))))  # o_O
cv_samples = np.array(cv_samples)
if len(cv_samples.shape) != 2 or cv_samples.shape[1] != 4:
    raise Exception(
        f"--cv-samples needs to be shaped (N, 4) but was {cv_samples.shape}"
    )
SampleDB().set_cv_values_from_npy(opts.run, cv_samples)
