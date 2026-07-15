import random
import argparse
import sys
import re
import numpy as np

# from util import *
from common.sample_db import SampleDB

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--run", type=str, required=True)
opts = parser.parse_args()

cv_samples = []
for line in sys.stdin.readlines():
    line_without_comment = re.sub(r"#.*", "", line.strip())
    if len(line_without_comment) == 0:
        continue
    cols = list(map(float, line_without_comment.split(",")))
    if len(cols) == 4:
        cv_samples.append(cols)
if len(cv_samples) == 0:
    raise Exception("ended up with no 4 col samples?")
cv_samples = np.array(cv_samples)
if len(cv_samples.shape) != 2 or cv_samples.shape[1] != 4:
    raise Exception(
        f"--cv-samples needs to be shaped (N, 4) but was {cv_samples.shape}"
    )
print("set", cv_samples.shape)

db = SampleDB()
db.delete_run(opts.run)
db.set_cv_values_from_npy(opts.run, cv_samples)
