#!/usr/bin/env bash
set -ex
cat sverilog_version/tests/network/net.out \
 | grep ^OUT\ 1 \
 | cut -b26-41 \
 | python3 single_width_bin_to_decimal.py \
 > y_pred.sverilog.txt
./plot.py --plot-png verilog.y_pred.png < y_pred.sverilog.txt
rm y_pred.sverilog.txt
geeqie verilog.y_pred.png
