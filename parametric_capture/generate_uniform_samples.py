import random
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .sampling import UniformSampler
from common.sample_db import SampleDB

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=str, required=True)
parser.add_argument("--delete", action="store_true", help="if set, delete old run")
parser.add_argument("--num-samples", type=int, required=True)
parser.add_argument("--seed", type=int, default=None, help="if none use int(--run)")
opts = parser.parse_args()

if opts.seed is None:
    seed = int(opts.run)
else:
    seed = opts.seed
print("seed", seed)


run_dir = Path("runs") / opts.run
run_dir.mkdir(parents=True, exist_ok=True)

# note: cv values bounded by (-1, 1) => (-10V, +10V)
# note: module only responds to (-0.6, 0.6 ) for a_cv and b_cv
bounds = []
bounds.append((-0.6, 0.6))  # a_cv
bounds.append((-0.6, 0.6))  # b_cv
bounds.append((-1, 1))  # morph
bounds.append((0.2, 0.8))  # ampltiude of multisine
uniform_sampler = UniformSampler(bounds=bounds, seed=seed)
samples = uniform_sampler.samples(num_samples=opts.num_samples)

print("first 10")
print(samples[:10])

db = SampleDB()
if opts.delete:
    db.delete_run(opts.run)
db.set_cv_values_from_npy(opts.run, samples)
