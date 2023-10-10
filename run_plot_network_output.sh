#!/usr/bin/env bash
set -ex
cat sverilog_version/tests/network/net.out \
 | grep ^OUT\ 1 \
 | cut -b26-41 \
 | python3 single_width_bin_to_decimal.py \
 > y_pred.sverilog.txt
./plot.py < y_pred.sverilog.txt
rm y_pred.sverilog.txt
geeqie plot.png
