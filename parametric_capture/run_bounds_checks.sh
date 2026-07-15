#!/usr/bin/env bash
set -ex
rm -rf runs/666 | true

cat > cv_samples.txt <<EOF
0.7, -0.7, -1.0, 0
0.7, -0.7, -0.75, 0
0.7, -0.7, -0.5, 0
0.7, -0.7, -0.25, 0
0.7, -0.7, 0, 0
0.7, -0.7, 0.25, 0
0.7, -0.7, 0.5, 0
0.7, -0.7, 0.75, 0
0.7, -0.7, 1, 0
EOF

uv run generate_explicit_samples.py --run 666 --cv-samples-txt cv_samples.txt
uv run capture.py --run 666 --explicitly-use-channels --sample-len-sec 1.0
uv run generate_plots.py --run 666
geeqie runs/666/plots