import random
import argparse
from pathlib import Path
from tqdm import tqdm
from ast import literal_eval

from audio_interface import AudioInterface
from sampling import SobolSampler
from plotting import *
from util import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=str, required=True)
parser.add_argument("--cv-samples-txt", type=Path, required=True)
opts = parser.parse_args()

run_dir = Path("runs") / opts.run
run_dir.mkdir(parents=True, exist_ok=True)

cv_samples = []
for line in open(opts.cv_samples_txt, "r").readlines():
    cv_samples.append(list(map(float, line.split(","))))  # o_O
cv_samples = np.array(cv_samples)
if len(cv_samples.shape) != 2 or cv_samples.shape[1] != 4:
    raise Exception(
        f"--cv-samples needs to be shaped (N, 4) but was {cv_samples.shape}"
    )
fname = run_dir / "cv_samples.npy"
np.save(fname, cv_samples)
print("saved cv_samples", cv_samples.shape, "to", fname)
