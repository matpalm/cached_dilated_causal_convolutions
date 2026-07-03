#!/usr/bin/env bash
set -ex

# dev testing set
#uv run generate_sobol_samples.py --run 000 --num-sobol-samples-po2 16
#uv run capture.py --run 000

# core sobol sampleset
# uv run generate_sobol_samples.py --run 001 --num-sobol-samples-po2 2048
# uv run capture.py --run 001
# uv run generate_model_data.py --run 001
# uv run generate_plots.py --run 001 --num 16

# extend sobol set
# uv run generate_sobol_samples.py \
#  --run 002 --num-sobol-samples-po2 2048 \
#  --seed 001 --fast-forward 2048
# uv run capture.py --run 002
# uv run generate_model_data.py --run 002
# uv run generate_plots.py --run 002 --num 16

# extend sobol set again
# uv run generate_sobol_samples.py \
#  --run 003 --num-sobol-samples-po2 4096 \
#  --seed 001 --fast-forward 4096
# uv run capture.py --run 003
# uv run generate_model_data.py --run 003
# uv run generate_plots.py --run 003 --num 16

# combine sobol sets into one
# uv run combine_runs.py --src 001 002 003 --dest 004
# uv run generate_model_data.py --run 004
# uv run generate_plots.py --run 004 --num 16

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

# loss based search
export KERAS_RUN=230_keras/i0
:> src_run.txt
echo 004 >> src_run.txt
for D in `seq 200 209`; do
   printf -v FD "%03d" $D
   uv run generate_candidates_by_loss.py \
     --src-run-file src_run.txt \
     --keras-run $KERAS_RUN \
     --dest-run $FD \
     --num-candidates 32 --density-weight 1
   uv run capture.py --run $FD
   uv run generate_model_data.py --run $FD
   pushd ..
   uv run -m keras_version.score_captures \
     --keras-run $KERAS_RUN \
     --model-data-z parametric_capture/runs/$FD/model_data.z/
   popd
   echo $FD >> src_run.txt
done
