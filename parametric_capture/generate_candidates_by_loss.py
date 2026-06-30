import numpy as np
from scipy.spatial import Delaunay
import zarr
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from util import *
from plotting import *
import pickle
from sklearn.preprocessing import MinMaxScaler

# seaborn just wont shut up
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--src-run",
    type=Path,
    help="where to read from",
    nargs="+",
    action="extend",
    default=[],
)
parser.add_argument("--src-run-file", type=Path, help="if set, add runs from this list")
parser.add_argument("--losses-tsv", type=Path)
parser.add_argument("--num-candidates", type=int, default=256)
parser.add_argument(
    "--density-weight",
    type=float,
    default=2.0,
    help="increase to penalise near cvs. 0 => no penalty",
)
parser.add_argument(
    "--dest-run", type=Path, required=True, help="where to write cv_samples.npy"
)
opts = parser.parse_args()
print(opts)

("runs" / opts.dest_run).mkdir(parents=True, exist_ok=True)

src_runs = []
for src_run in opts.src_run:
    src_runs.append(src_run)
if opts.src_run_file and opts.src_run_file.exists():
    with open(opts.src_run_file, "r") as f:
        for line in f.readlines():
            src_runs.append(Path(line.strip()))

# load loss values for each run
records = []
for src_run in src_runs:
    with open("runs" / src_run / opts.losses_tsv) as f:
        for line_num, line in enumerate(f.readlines()):
            if line.startswith("mse"):
                continue
            mse, huber, sftf = map(float, line.split("\t"))
            records.append({"mse": mse, "huber": huber, "sftf": sftf})
losses_df = pd.DataFrame(records)
print(losses_df.describe())
del records
print("|losses|", len(losses_df))

cv_samples = []
for src_run in src_runs:
    cv_samples.append(np.load("runs" / src_run / "cv_samples.npy"))
cv_samples = np.vstack(cv_samples)
print("cv_samples", cv_samples.shape)

assert len(losses_df) == len(cv_samples)

tri = Delaunay(cv_samples)
num_simplex_vertices = tri.simplices.shape[-1]
print("num_simplex_vertices", num_simplex_vertices)

edges = set()
for simplex in tri.simplices:
    for i in range(num_simplex_vertices):
        for j in range(i + 1, num_simplex_vertices):
            edge = tuple(sorted((simplex[i], simplex[j])))
            edges.add(edge)
unique_edges = list(edges)
print("|unique_edges|", len(unique_edges))


# calculate all edge midpoints and lengths
# do this pass first to get density_scores
edge_midpoints = []
edge_lengths = []
for edge in unique_edges:
    pi, pj = edge
    cv_i, cv_j = cv_samples[pi], cv_samples[pj]
    edge_midpoints.append((cv_i + cv_j) / 2.0)
    edge_lengths.append(np.linalg.norm(cv_i - cv_j))
edge_midpoints = np.array(edge_midpoints)
edge_lengths = np.array(edge_lengths)

# from midpoints calculate edge density_scores
# ( base on dynamic radius. is 0.6 too relaxed? )
radius = np.mean(edge_lengths) * 0.6
nn = NearestNeighbors(radius=0.1).fit(cv_samples)
density_counts = nn.radius_neighbors(edge_midpoints, return_distance=False)
densities = np.array([len(neighbors) for neighbors in density_counts], dtype=float)
# density_weight = 2.0  # higher => pushed apart more
# density_scores = 1.0 + density_weight * densities
# print("density_scores", density_scores.shape)
# print("densities", list(densities))

# scaling scoring pieces
scaler = MinMaxScaler()
for col in ["mse", "huber", "sftf"]:
    losses_df[col] = scaler.fit_transform(losses_df[[col]])

hubers = np.array(losses_df["huber"])
sftfs = np.array(losses_df["sftf"])

records = []
for e, edge in enumerate(unique_edges):
    record = {}

    # record pts
    pi, pj = edge
    record["pi"] = pi
    record["pj"] = pj

    # losses
    record["mean_huber"] = (hubers[pi] + hubers[pj]) / 2
    record["mean_sftf"] = (sftfs[pi] + sftfs[pj]) / 2
    record["loss_sum"] = record["mean_huber"] + record["mean_sftf"]

    # local density
    local_density = densities[e]
    record["local_density"] = local_density

    # combined overall score
    score = record["loss_sum"] / (1.0 + (opts.density_weight * local_density))
    record["score"] = score

    records.append(record)

edge_scores_df = pd.DataFrame(records)

# write some stats to a file ( then stdout )
with open("runs" / opts.dest_run / "candidate_generation_stats.txt", "w") as f:
    print("opts", opts, file=f)
    print("src_runs", list(map(str, src_runs)), file=f)
    for col in ["loss_sum", "local_density", "score"]:
        print("top by", col, file=f)
        print(edge_scores_df.sort_values(col, ascending=False).head(10), file=f)
with open("runs" / opts.dest_run / "candidate_generation_stats.txt", "r") as f:
    print(f.read())

# write candidates
topN_scores = edge_scores_df.sort_values("score", ascending=False).head(
    opts.num_candidates
)
topN_pi = list(topN_scores["pi"])
topN_pj = list(topN_scores["pj"])
candidate_cvs = []
for pi, pj in zip(topN_pi, topN_pj):
    cv_i, cv_j = cv_samples[pi], cv_samples[pj]
    candidate_cvs.append((cv_i + cv_j) / 2)
candidate_cvs = np.stack(candidate_cvs)

fname = "runs" / opts.dest_run / "cv_samples.npy"
print("wrote", fname, candidate_cvs.shape)
np.save(fname, candidate_cvs)
