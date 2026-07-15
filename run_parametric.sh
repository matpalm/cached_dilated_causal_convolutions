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

# extend sobol set again
# uv run -m parametric_capture.generate_sobol_samples \
#  --run 005 --num-sobol-samples-po2 8192 \
#  --seed 001 --fast-forward 8192
# uv run -m parametric_capture.capture --run 005
# uv run -m parametric_capture.generate_model_data --run 005
# uv run -m parametric_capture.combine_runs --srcs 004 005 --dest 006


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

export SOBOL_SET=006

# first step is a bootstrap

# 1) initial run on sobol data to get model for bootstrapped scoring
# uv run -m keras_version.train \
#  --run 232_keras/i0 \
#  --capture-run $SOBOL_SET \
#  --alpha-mse 1.0 --use-huber-loss --beta-stft 0.01 --beta-stft-warmup 0.25 --beta-stft-ramp 0.25 \
#  --batch-size 64 --num-train-batches 200 --epochs 10

# 2) score sobol samples for guiding hard example generation
# uv run -m keras_version.score_captures \
#  --keras-run 232_keras/i0 \
#  --capture-runs $SOBOL_SET

# 3) run first round of importance sampling
# export KERAS_RUN=232_keras/i0
# :> src_run.txt
# echo $SOBOL_SET >> src_run.txt
# for D in `seq 201 209`; do
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
# uv run -m parametric_capture.combine_runs \
#  --srcs `seq -s" " 201 209` \
#  --dest 210

# --------------------------------
# iterative importance sampling runs

importance_sampling_iteration() {

    export NEW_HARD_FROM=$(( $HARD_EGS_SET + 1))
    export NEW_HARD_TO=$(( $HARD_EGS_SET + 19))
    export NEW_HARD_EGS_SET=$(( $HARD_EGS_SET + 20))

    :> src_run.txt
    echo $SOBOL_SET >> src_run.txt
    echo $HARD_EGS_SET >> src_run.txt
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
      --srcs $HARD_EGS_SET `seq -s" " $NEW_HARD_FROM $NEW_HARD_TO` \
      --dest $NEW_HARD_EGS_SET

    # importance sampling training
    uv run -m keras_version.train_is \
      --run $NEW_KERAS_RUN --restore-run $KERAS_RUN \
      --sobol-capture-run 006 \
      --hard-capture-run $NEW_HARD_EGS_SET --keras-model $KERAS_RUN \
      --alpha-mse 1.0 --use-huber-loss --beta-stft 0.01 \
      --beta-stft-warmup $RAMP_WARMUP --beta-stft-ramp $RAMP_WARMUP \
      --batch-size 64 --hard-batch-egs 8 \
      --train-batches-per-epoch 200 \
      --epochs $EPOCHS
    uv run -m keras_version.score_captures \
      --keras-run $NEW_KERAS_RUN \
      --capture-runs $SOBOL_SET $NEW_HARD_EGS_SET

    uv run -m parametric_capture.check_chunk_sizes --src $NEW_HARD_EGS_SET


}

# export KERAS_RUN=232_keras/i0
# export HARD_EGS_SET=210
# export NEW_KERAS_RUN=232_keras/i1
# export EPOCHS=10
# export RAMP_WARMUP=0.25
# importance_sampling_iteration

# export KERAS_RUN=232_keras/i1
# export HARD_EGS_SET=230
# export NEW_KERAS_RUN=232_keras/i2
# export EPOCHS=10
# export RAMP_WARMUP=0.25
# importance_sampling_iteration

# export KERAS_RUN=232_keras/i2
# export HARD_EGS_SET=250
# export NEW_KERAS_RUN=232_keras/i3
# export EPOCHS=10
# export RAMP_WARMUP=0.25
# importance_sampling_iteration

# export KERAS_RUN=232_keras/i3
# export HARD_EGS_SET=270
# export NEW_KERAS_RUN=232_keras/i4
# export EPOCHS=10
# export RAMP_WARMUP=0.25
# importance_sampling_iteration

# export KERAS_RUN=232_keras/i4
# export HARD_EGS_SET=290
# export NEW_KERAS_RUN=232_keras/i5
# export EPOCHS=10
# export RAMP_WARMUP=0.25
# importance_sampling_iteration

# export KERAS_RUN=232_keras/i5
# export HARD_EGS_SET=310
# export NEW_KERAS_RUN=232_keras/i6
# export EPOCHS=15
# export RAMP_WARMUP=0
# importance_sampling_iteration

# export KERAS_RUN=232_keras/i6
# export HARD_EGS_SET=330
# export NEW_KERAS_RUN=232_keras/i7
# export EPOCHS=20
# export RAMP_WARMUP=0
# importance_sampling_iteration

# export KERAS_RUN=232_keras/i7
# export HARD_EGS_SET=350
# export NEW_KERAS_RUN=232_keras/i8
# export EPOCHS=25
# export RAMP_WARMUP=0
# importance_sampling_iteration

# export KERAS_RUN=232_keras/i8
# export HARD_EGS_SET=370
# export NEW_KERAS_RUN=232_keras/i9
# export EPOCHS=30
# export RAMP_WARMUP=0
# importance_sampling_iteration

export FINAL_IS_SET=390
export FINAL_KERAS_RUN=232_keras/i9

#----------------------------------------
# run capture over uniform sampling over cv_values
# a_cv & b_cv => [-0.6, 0.6] morph => [-1, 1]

export UNIFORM_SET=500
# uv run -m parametric_capture.generate_uniform_samples --run $R --delete --num-samples 5000
# uv run -m parametric_capture.capture --run $R
# uv run -m parametric_capture.generate_model_data --run $R
# uv run -m parametric_capture.generate_plots --run $R --num 20
#uv run -m keras_version.score_captures --keras-run $FINAL_KERAS_RUN --capture-runs $UNIFORM_SET

uv run -m common.sample_db --delete --run 600
rm -rf parametric_capture/runs/600
uv run -m keras_version.add_y_pred_teacher \
  --keras-run $FINAL_KERAS_RUN \
  --src-runs $SOBOL_RUN $FINAL_IS_SET $UNIFORM_SET \
  --dest-run 600

