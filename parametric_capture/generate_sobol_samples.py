import random
import argparse
from pathlib import Path
from tqdm import tqdm

from audio_interface import AudioInterface
from sampling import *
from plotting import *
from util import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=str, required=True)
parser.add_argument(
    "--num-sobol-samples-po2", type=int, default=4, help="should be po2"
)
parser.add_argument("--seed", type=int, default=None, help="if none use int(--run)")
opts = parser.parse_args()

if opts.seed is None:
    seed = int(opts.run)
else:
    seed = opts.seed
print("seed", seed)

run_dir = Path("runs") / opts.run
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "plots").mkdir(parents=True, exist_ok=True)
(run_dir / "cv_buffers").mkdir(parents=True, exist_ok=True)
(run_dir / "capture_buffers").mkdir(parents=True, exist_ok=True)

# get initial sobol samples for 3 CV values and A
# note: cv values bounded by (-1, 1) => (-10V, +10V)
#       we use 0.75 instead of 1.0 to avoid very low amp multisines at edges
bounds = []
bounds.append((-1, 1))
bounds.append((-1, 1))
bounds.append((-1, 1))
bounds.append((-1, 1))
bounds.append((0.2, 0.9))  # ampltiude
sobol_sampler = SobolSampler(bounds=bounds, seed=seed)
samples = sobol_sampler.samples(num_samples_po2=opts.num_sobol_samples_po2)
fname = run_dir / "cv_samples.npy"
np.save(fname, samples)
print("wrote", fname, samples.shape)
