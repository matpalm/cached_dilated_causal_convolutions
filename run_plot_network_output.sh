#!/usr/bin/env bash
if [[ -z "$WAVE" ]]; then
  echo "WAVE not set"
  exit 1
fi
set -ex
cat sverilog_version/tests/network/net.$WAVE.out \
 | grep "^OUT dec" | grep -v xxxx | cut -f3 -d' ' | uniq \
 > y_pred.sverilog.$WAVE.txt
./plot.py --plot-png verilog.y_pred.$WAVE.png < y_pred.sverilog.$WAVE.txt
rm y_pred.sverilog.$WAVE.txt
geeqie verilog.y_pred.$WAVE.png
