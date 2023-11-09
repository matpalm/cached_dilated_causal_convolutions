#!/usr/bin/env bash
if [[ -z "$WAVE" ]]; then
  echo "WAVE not set"
  exit 1
fi

if [[ -z "$RUN" ]]; then
  echo "RUN not set"
  exit 1
fi

set -x

RUN_DIR=`pwd`/runs/$RUN/

# clear old runs ( since they can have cached params )
pushd sverilog_version/tests/network/
rm -rf sim_build results.xml

# run iverilog sim
rm test_x.hex
ln -s $RUN_DIR/test_x_files/test_x.$WAVE.hex test_x.hex
rm -r weights
ln -s $RUN_DIR/weights/verilog/latest/ weights
make > $RUN_DIR/net.$WAVE.out
popd

# generate plot, TODO: put these onto one plot
cat $RUN_DIR/net.$WAVE.out \
 | grep "^OUT dec" | cut -f3 -d' ' | grep -v xxxx | uniq \
 > /tmp/$$.y_pred.sverilog.$WAVE.ssv
cat /tmp/$$.y_pred.sverilog.$WAVE.ssv | ./plot.py --plot-png $RUN_DIR/verilog.y_pred.$WAVE.png
rm /tmp/$$.y_pred.sverilog.$WAVE.ssv
