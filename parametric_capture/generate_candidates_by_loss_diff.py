import argparse
import numpy as np
from scipy.spatial import Delaunay
import zarr
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MinMaxScaler

from common.sample_db import SampleDB
from common.loss_cache import CachedEdgeLoss

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
# parser.add_argument("--keras-run", type=str, help="for losses in db", required=True)
parser.add_argument("--num-candidates", type=int, default=256)
parser.add_argument(
    "--density-weight",
    type=float,
    default=2.0,
    help="increase to penalise near cvs. 0 => no penalty",
)
parser.add_argument(
    "--dest-run", type=Path, required=True, help="where to write stats on candidates"
)
parser.add_argument(
    "--alpha-huber",
    type=float,
    default=1.0,
    help="--alpha-mse ( huber ) from converged keras model",
)
parser.add_argument(
    "--beta-stft",
    type=float,
    default=0.01,
    help="--beta-stft from converged keras model",
)
opts = parser.parse_args()
print(opts)

("runs" / opts.dest_run).mkdir(parents=True, exist_ok=True)

db = SampleDB()
cached_edge_loss = CachedEdgeLoss()

src_runs = []
for src_run in opts.src_run:
    src_runs.append(src_run)
if opts.src_run_file and opts.src_run_file.exists():
    with open(opts.src_run_file, "r") as f:
        for line in f.readlines():
            src_runs.append(Path(line.strip()))

# # load loss values for each run
# loss_rows = []
# for src_run in src_runs:
#     losses = db.losses_for(src_run, model=opts.keras_run)
#     print("src_run", src_run, "model", opts.keras_run, "|losses|", len(losses))
#     loss_rows.extend(losses)
# losses_df = pd.DataFrame(loss_rows)
# del loss_rows
# print(losses_df.describe())
# print("|losses|", len(losses_df))
# assert len(losses_df) > 0

cv_samples = []
cv_sample_idx_to_run_and_idx = []
for src_run in src_runs:
    cv_values = db.cv_values_for(src_run)
    cv_samples.append(cv_values)
    for i in range(len(cv_values)):
        cv_sample_idx_to_run_and_idx.append((str(src_run), i))
cv_samples = np.vstack(cv_samples)
print("cv_samples", cv_samples.shape)
# if len(losses_df) != len(cv_samples):
#     raise Exception(f"|losses_df| {len(losses_df)} != |cv_samples| {len(cv_samples)}")

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
# scaler = MinMaxScaler()
# for col in ["huber", "stft"]:
#     losses_df[col] = scaler.fit_transform(losses_df[[col]])
# hubers = np.array(losses_df["huber"])
# stft = np.array(losses_df["stft"])


records = []
for e, edge in enumerate(tqdm(unique_edges)):
    record = {}

    # record pts
    pi, pj = edge
    record["pi"] = pi
    record["pj"] = pj

    # mark points in dataframe as run and idx
    run_i, idx_i = cv_sample_idx_to_run_and_idx[pi]
    run_j, idx_j = cv_sample_idx_to_run_and_idx[pj]
    record["run_i"] = f"{run_i}_{idx_i}"
    record["run_j"] = f"{run_j}_{idx_j}"

    # cv distance
    cv_dist = edge_lengths[e]
    record["cv_dist"] = cv_dist

    # losses
    _loss, huber, stft = cached_edge_loss.get(run_i, idx_i, run_j, idx_j)
    record["huber"] = huber  # just to show
    record["stft"] = stft
    record["loss"] = stft

    # gradient as spectral loss / cv distance
    # high value => large change for small cv diff
    local_grad = stft / cv_dist
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
    print("opts", opts, file=f)
    print("src_runs", list(map(str, src_runs)), file=f)
    for col in ["loss", "local_density", "score"]:
        print("----------- top by", col, file=f)
        print(edge_scores_df.sort_values(col, ascending=False).head(10), file=f)
        print("----------- bottom by", col, file=f)
        print(edge_scores_df.sort_values(col, ascending=True).head(10), file=f)
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
db.set_cv_values_from_npy(opts.dest_run, candidate_cvs)
