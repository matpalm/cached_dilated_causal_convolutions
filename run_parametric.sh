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

# --------------------------------
# importance sampling run

# initial short run on sobol data to get model for scoring
# uv run -m keras_version.train \
#  --run 231_keras/i0 \
#  --capture-run 004 \
#  --alpha-mse 1.0 --use-huber-loss --beta-stft 0.01 \
#  --batch-size 64 --num-train-batches 1000 --epochs 1

# score sobol samples for guiding hard example generation
# uv run -m keras_version.score_captures \
#  --keras-run 231_keras/i0 \
#  --capture-runs 004

# loss based search; round 1
# export KERAS_RUN=231_keras/i0
# :> src_run.txt
# echo 004 >> src_run.txt
# for D in `seq 200 209`; do
#    printf -v FD "%03d" $D
#    uv run -m parametric_capture.generate_candidates_by_loss \
#      --src-run-file src_run.txt \
#      --keras-run $KERAS_RUN \
#      --dest-run $FD \
#      --num-candidates 32 --density-weight 1
#    uv run -m parametric_capture.capture --run $FD
#    uv run -m parametric_capture.generate_model_data --run $FD
#    uv run -m keras_version.score_captures \
#      --keras-run $KERAS_RUN \
#      --capture-runs $FD
#    echo $FD >> src_run.txt
# done

# combine runs
# uv run -m parametric_capture.combine_runs \
#  --srcs `seq -s" " 200 209` \
#  --dest 210

# PLOT 3d vis of cv_values for 210


# run importance sampling using sobol and these hard examples
# uv run -m keras_version.train_is \
#  --run 231_keras/i1 --restore-run 231_keras/i0 \
#  --sobol-capture-run 004 \
#  --hard-capture-run 210 --keras-model 231_keras/i0 \
#  --alpha-mse 1.0 --use-huber-loss --beta-stft 0.01 --beta-stft-warmup 0.25 --beta-stft-ramp 0.25 \
#  --batch-size 64 --hard-batch-egs 8 \
#  --train-batches-per-epoch 100 \
#  --epochs 10

# uv run -m keras_version.score_captures \
#  --keras-run 231_keras/i1 \
#  --capture-runs 004 210

# PLOT sobol/hard huber/stft runs plot

# generate new candidates by loss and cv_space density and capture them
# export KERAS_RUN=231_keras/i1
# :> src_run.txt
# echo 004 >> src_run.txt
# echo 210 >> src_run.txt
# for D in `seq 220 229`; do
#    printf -v FD "%03d" $D
#    uv run -m parametric_capture.generate_candidates_by_loss \
#      --src-run-file src_run.txt \
#      --keras-run $KERAS_RUN \
#      --dest-run $FD \
#      --num-candidates 32 --density-weight 1
#    uv run -m parametric_capture.capture --run $FD
#    uv run -m parametric_capture.generate_model_data --run $FD
#    uv run -m keras_version.score_captures \
#      --keras-run $KERAS_RUN \
#      --capture-runs $FD
#    echo $FD >> src_run.txt
# done

# combine runs 210 + last => 640 egs
# uv run -m parametric_capture.combine_runs \
#  --srcs 210 `seq -s" " 220 229` \
#  --dest 230

# PLOT 3d vis of cv_values for 220 to 229   ( ie without 210 )

# second round of importance sampling training
# uv run -m keras_version.train_is \
#  --run 231_keras/i2 --restore-run 231_keras/i1 \
#  --sobol-capture-run 004 \
#  --hard-capture-run 230 --keras-model 231_keras/i1 \
#  --alpha-mse 1.0 --use-huber-loss --beta-stft 0.01 --beta-stft-warmup 0.25 --beta-stft-ramp 0.25 \
#  --batch-size 64 --hard-batch-egs 8 \
#  --train-batches-per-epoch 100 \
#  --epochs 10

# uv run -m keras_version.score_captures \
#  --keras-run 231_keras/i2 \
#  --capture-runs 004 230

# though this was never trained it's useful for graphs to
# see the original loss ( i0 ) before any IS training on
# the 'hard examples' we have
# uv run -m keras_version.score_captures \
#  --keras-run 231_keras/i0 \
#  --capture-runs 230

# PLOT sobol/hard huber/stft runs plot
:> src_run.txt
echo 004 >> src_run.txt
#echo 230 >> src_run.txt
uv run -m parametric_capture.generate_candidates_by_loss_diff \
    --src-run-file src_run.txt \
    --dest-run 666 \
    --num-candidates 32 --density-weight 1


