#!/usr/bin/env bash
set -ex

# use loss diff between edge points to guide sampling hard
# i.e. large loss diff / small cv diff => large local gradient
#        => area of interest

:> src_run.txt
echo 006 >> src_run.txt
for D in `seq 501 600`; do
  printf -v FD "%03d" $D
  uv run -m parametric_capture.generate_candidates_by_loss_diff \
    --src-run-file src_run.txt \
    --dest-run $FD \
    --num-candidates 32 \
    --density-weight 1
  uv run -m parametric_capture.capture --run $FD
  uv run -m parametric_capture.generate_model_data --run $FD
  echo $FD >> src_run.txt
done
