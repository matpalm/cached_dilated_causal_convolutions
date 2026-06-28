import numpy as np
from scipy.spatial import Delaunay
import zarr
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from util import *
from plotting import *

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

records = []
for src_run in src_runs:
    capture_buffers_zarr = zarr.open("runs" / src_run / "capture_buffers.z", mode="r")
    # print("capture_buffers_zarr nchunks", capture_buffers_zarr.nchunks)
    for b in range(capture_buffers_zarr.nchunks):
        sample = capture_buffers_zarr.blocks[b]
        ch0_of_sample = sample[:, 0]
        stats = calculate_audio_stats(ch0_of_sample, ignore_in_out=500)
        records.append(stats)
# print("capture_buffers_zarr ch0")
capture_buffers_df = pd.DataFrame(records)
print(capture_buffers_df.describe())
del records

cv_samples = []
for src_run in src_runs:
    cv_samples.append(np.load("runs" / src_run / "cv_samples.npy"))
cv_samples = np.vstack(cv_samples)
print("cv_samples", cv_samples.shape)

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


def calculate_local_grads(audio_stat):
    local_grads = []
    for edge in unique_edges:
        pi, pj = edge
        cv_i, cv_j = cv_samples[pi], cv_samples[pj]
        cv_distance = np.linalg.norm(cv_i - cv_j)
        # print('cv_i cv_j', cv_i, cv_j, 'dist', cv_distance)
        stat_i, stat_j = audio_stat[pi], audio_stat[pj]
        stat_distance = np.abs(stat_i - stat_j)
        # print('stat_i, stat_j', stat_i, stat_j, 'dist', stat_distance)
        local_grad = stat_distance / cv_distance
        local_grads.append((local_grad, stat_distance, cv_distance, pi, pj))
    return pd.DataFrame(
        local_grads, columns="local_grad stat_distance cv_distance pi pj".split(" ")
    )


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

# calculate everything else
spectral_centroid = min_max_scale(np.array(capture_buffers_df["spectral_centroid"]))
odd_even = min_max_scale(np.array(capture_buffers_df["odd_even"]))
records = []
for e, edge in enumerate(unique_edges):
    record = {}

    # record pts
    pi, pj = edge
    record["pi"] = pi
    record["pj"] = pj

    # cv distance
    cv_dist = edge_lengths[e]
    record["cv_dist"] = cv_dist

    # stats
    spectral_centroid_dist = np.abs(spectral_centroid[pi] - spectral_centroid[pj])
    record["spectral_centroid_dist"] = spectral_centroid_dist
    odd_even_dist = np.abs(odd_even[pi] - odd_even[pj])
    record["odd_even_dist"] = odd_even_dist

    # gradient as delta_stat / cv distance
    local_grad = (spectral_centroid_dist + odd_even_dist) / cv_dist
    record["local_grad"] = local_grad

    # local density
    local_density = densities[e]
    record["local_density"] = local_density

    # combined overall score
    score = local_grad / (1.0 + (opts.density_weight * local_density))
    record["score"] = score

    records.append(record)

edge_scores_df = pd.DataFrame(records)

# write some stats to a file ( then stdout )
with open("runs" / opts.dest_run / "candidate_generation_stats.txt", "w") as f:
    for col in ["local_grad", "local_density", "score"]:
        print("top by", col, file=f)
        print(edge_scores_df.sort_values(col, ascending=False).head(), file=f)
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
