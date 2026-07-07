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
#  --srcs `seq -s" " 220 229` \
#  --dest 220_229
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

# generate new candidates by loss and cv_space density and capture them
# do 20 runs, not just 10 this time
# export KERAS_RUN=231_keras/i2
# :> src_run.txt
# echo 004 >> src_run.txt
# echo 230 >> src_run.txt
# for D in `seq 240 259`; do
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
#   --srcs `seq -s" " 240 259` \
#   --dest 240_259
# uv run -m parametric_capture.combine_runs \
#   --srcs 230 `seq -s" " 240 259` \
#   --dest 260

# third round of importance sampling training
#  - no beta warmup/ramp
#  - 200 epochs
# uv run -m keras_version.train_is \
#  --run 231_keras/i3 --restore-run 231_keras/i2 \
#  --sobol-capture-run 004 \
#  --hard-capture-run 260 --keras-model 231_keras/i2 \
#  --alpha-mse 1.0 --use-huber-loss --beta-stft 0.01 --beta-stft-warmup 0 --beta-stft-ramp 0 \
#  --batch-size 64 --hard-batch-egs 8 \
#  --train-batches-per-epoch 200 \
#  --epochs 10

# uv run -m keras_version.score_captures \
#  --keras-run 231_keras/i3 \
#  --capture-runs 004 260

# PLOT sobol/hard huber/stft runs plot

# generate new candidates by loss and cv_space density and capture them
# increase from 20 to 30 capture run loops
# also drop density weight from 1.0 to 0.1
# export KERAS_RUN=231_keras/i3
# export HARD_EGS_RUN=260
# :> src_run.txt
# echo 004 >> src_run.txt
# echo $HARD_EGS_RUN >> src_run.txt
# for D in `seq 270 299`; do
#    printf -v FD "%03d" $D
#    uv run -m parametric_capture.generate_candidates_by_loss \
#      --src-run-file src_run.txt \
#      --keras-run $KERAS_RUN \
#      --dest-run $FD \
#      --num-candidates 32 --density-weight 0.1
#    uv run -m parametric_capture.capture --run $FD
#    uv run -m parametric_capture.generate_model_data --run $FD
#    uv run -m keras_version.score_captures \
#      --keras-run $KERAS_RUN \
#      --capture-runs $FD
#    echo $FD >> src_run.txt
# done

# # combine runs 210 + last => 640 egs
# export NEW_HARD_EGS_RUN=300
# uv run -m parametric_capture.combine_runs \
#   --srcs `seq -s" " 270 299` \
#   --dest 270_299
# uv run -m parametric_capture.combine_runs \
#   --srcs $HARD_EGS_RUN `seq -s" " 270 299` \
#   --dest $NEW_HARD_EGS_RUN

# # 4th round of importance sampling training
# #  - no beta warmup/ramp
# #  - 300 epochs
# export NEW_KERAS_RUN=231_keras/i4
# uv run -m keras_version.train_is \
#  --run $NEW_KERAS_RUN --restore-run $KERAS_RUN \
#  --sobol-capture-run 004 \
#  --hard-capture-run $NEW_HARD_EGS_RUN --keras-model $KERAS_RUN \
#  --alpha-mse 1.0 --use-huber-loss --beta-stft 0.01 --beta-stft-warmup 0 --beta-stft-ramp 0 \
#  --batch-size 64 --hard-batch-egs 8 \
#  --train-batches-per-epoch 300 \
#  --epochs 10
# uv run -m keras_version.score_captures \
#  --keras-run $NEW_KERAS_RUN \
#  --capture-runs 004 $NEW_HARD_EGS_RUN

# PLOT sobol/hard huber/stft runs plot

importance_sampling_iteration() {

    export NEW_HARD_FROM=$(( $HARD_EGS_RUN + 1))
    export NEW_HARD_TO=$(( $HARD_EGS_RUN + 19))
    export NEW_HARD_EGS_RUN=$(( $HARD_EGS_RUN + 20))

    :> src_run.txt
    echo 004 >> src_run.txt
    echo $HARD_EGS_RUN >> src_run.txt
    for D in `seq $NEW_HARD_FROM $NEW_HARD_TO`; do
      printf -v FD "%03d" $D
      uv run -m parametric_capture.generate_candidates_by_loss \
          --src-run-file src_run.txt \
          --keras-run $KERAS_RUN \
          --dest-run $FD \
          --num-candidates 32 --density-weight 0.1
      uv run -m parametric_capture.capture --run $FD
      uv run -m parametric_capture.generate_model_data --run $FD
      uv run -m keras_version.score_captures \
          --keras-run $KERAS_RUN \
          --capture-runs $FD
      echo $FD >> src_run.txt
    done

    uv run -m parametric_capture.combine_runs \
      --srcs `seq -s" " $NEW_HARD_FROM $NEW_HARD_TO` \
      --dest ${NEW_HARD_FROM}_${NEW_HARD_TO}
    uv run -m parametric_capture.combine_runs \
      --srcs $HARD_EGS_RUN `seq -s" " $NEW_HARD_FROM $NEW_HARD_TO` \
      --dest $NEW_HARD_EGS_RUN
    uv run -m common.sample_db \
      --delete \
      --run `seq -s" " $NEW_HARD_FROM $NEW_HARD_TO`

    # # 4th round of importance sampling training
    # #  - no beta warmup/ramp
    # #  - 300 epochs
    uv run -m keras_version.train_is \
      --run $NEW_KERAS_RUN --restore-run $KERAS_RUN \
      --sobol-capture-run 004 \
      --hard-capture-run $NEW_HARD_EGS_RUN --keras-model $KERAS_RUN \
      --alpha-mse 1.0 --use-huber-loss --beta-stft 0.01 --beta-stft-warmup 0 --beta-stft-ramp 0 \
      --batch-size 64 --hard-batch-egs 8 \
      --train-batches-per-epoch 300 \
      --epochs 10
    uv run -m keras_version.score_captures \
      --keras-run $NEW_KERAS_RUN \
      --capture-runs 004 $NEW_HARD_EGS_RUN

}

# export KERAS_RUN=231_keras/i6
# export HARD_EGS_RUN=340
# export NEW_KERAS_RUN=231_keras/i7
# importance_sampling_iteration

# export KERAS_RUN=231_keras/i7
# export HARD_EGS_RUN=360
# export NEW_KERAS_RUN=231_keras/i8
# importance_sampling_iteration

# export KERAS_RUN=231_keras/i7
# export HARD_EGS_RUN=380
# export NEW_KERAS_RUN=231_keras/i8
# importance_sampling_iteration

# final IS run => 400

# extend sobol set again
# uv run -m parametric_capture.generate_sobol_samples \
#  --run 005 --num-sobol-samples-po2 8192 \
#  --seed 001 --fast-forward 8192
# uv run -m parametric_capture.capture --run 005
#uv run -m parametric_capture.generate_model_data --run 005
#uv run -m parametric_capture.combine_runs --srcs 004 005 --dest 006


# finally a uniform sampling over cv_values
# a_cv & b_cv => [-0.6, 0.6] morph => [-1, 1]
export R=500
#uv run -m parametric_capture.generate_uniform_samples --run $R --delete --num-samples 5000
uv run -m parametric_capture.capture --run $R
uv run -m parametric_capture.generate_model_data --run $R
uv run -m parametric_capture.generate_plots --run $R --num 20
