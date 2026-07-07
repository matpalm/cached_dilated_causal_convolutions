import random
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from common.sample_db import SampleDB

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=str, required=True)
parser.add_argument("--delete", action="store_true", help="if set, delete old run")
parser.add_argument("--step-size", type=float, default=0.1)
opts = parser.parse_args()

run_dir = Path("runs") / opts.run
run_dir.mkdir(parents=True, exist_ok=True)


def stepped_lin_space(a, b, step):
    assert a < b
    n = int((b - a) / step) + 1
    return np.linspace(a, b, num=n)


# recall; module only responds to -0.6, 0.6 for a_cv & b_cv
cv_values = []
for a_cv in stepped_lin_space(-0.6, 0.6, opts.step_size):
    for b_cv in stepped_lin_space(-0.6, 0.6, opts.step_size):
        for morph_cv in stepped_lin_space(-1, 1, opts.step_size):
            cv_values.append((a_cv, b_cv, morph_cv, 0.5))

cv_values = np.array(cv_values)
print("cv_values[:10]", cv_values[:10])
print("cv_values[-10:]", cv_values[-10:])
print("cv_values", cv_values.shape)

db = SampleDB()
if opts.delete:
    db.delete_run(opts.run)
db.set_cv_values_from_npy(opts.run, cv_values)
