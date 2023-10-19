#!/usr/bin/env bash
if [[ -z "$WAVE" ]]; then
  echo "WAVE not set"
  exit 1
fi

set -x
rm -rf sverilog_version/tests/network/{weights,net.out,test_x.hex} y_pred.sverilog.txt verilog.y_pred.png

set -e
cp -r weights sverilog_version/tests/network/weights
cp test_x*hex sverilog_version/tests/network/

cd sverilog_version/tests/network
ln -s test_x.$WAVE.hex test_x.hex
make | tee net.$WAVE.out