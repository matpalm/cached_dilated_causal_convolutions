#!/usr/bin/env bash
if [[ -z "$WAVE" ]]; then
  echo "WAVE not set"
  exit 1
fi

set -x
rm -rf sverilog_version/tests/network/{net.out,test_x.hex} y_pred.sverilog.txt verilog.y_pred.png

set -e
cp test_x*hex sverilog_version/tests/network/

# run iverilog sim
pushd sverilog_version/tests/network
ln -s test_x.$WAVE.hex test_x.hex
make | tee net.$WAVE.out
popd

# generate plot
cat sverilog_version/tests/network/net.$WAVE.out \
 | grep "^OUT dec" | cut -f3 -d' ' | grep -v xxxx | uniq \
 > y_pred.sverilog.$WAVE.txt
./plot.py --plot-png verilog.y_pred.$WAVE.png < y_pred.sverilog.$WAVE.txt
rm y_pred.sverilog.$WAVE.txt
