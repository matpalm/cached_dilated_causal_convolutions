#!/usr/bin/env bash
set -ex

# dev testing set
#uv run generate_sobol_samples.py --run 000 --num-sobol-samples-po2 16
#uv run capture.py --run 000

# core sobol sampleset
# uv run generate_sobol_samples.py --run 001 --num-sobol-samples-po2 2048
# uv run capture.py --run 001
# uv run generate_plots.py --run 001 --num 16

# local_grad guided search
# no density weighting ( to demonstrate fixation )
# 10 iterations
# echo 001 > src_run.txt
# for D in `seq 10 19`; do
#   printf -v FD "%03d" $D
#   uv run generate_candidates.py \
#     --src-run-file src_run.txt --dest-run $FD \
#     --num-candidates 32 --density-weight 0
#   uv run capture.py --run $FD
#   echo $FD >> src_run.txt
# done

# # local_grad guided search
# # with density weighting ( to demonstrate exploration )
# # 100 iterations
# echo 001 > src_run.txt
# for D in `seq 20 119`; do
#   printf -v FD "%03d" $D
#   uv run generate_candidates.py \
#     --src-run-file src_run.txt --dest-run $FD \
#     --num-candidates 32 --density-weight 20
#   uv run capture.py --run $FD
#   echo $FD >> src_run.txt
# done

# extend sobol set 001 -> 002
uv run generate_sobol_samples.py \
 --run 002 --num-sobol-samples-po2 2048 \
 --seed 001 --fast-forward 2048
uv run capture.py --run 002