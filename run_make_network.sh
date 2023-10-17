#!/usr/bin/env bash
set -x
rm sverilog_version/tests/network/net.out y_pred.sverilog.txt verilog.y_pred.png
set -e
cd ~/dev/cached_dilated_causal_convolutions/sverilog_version/tests/network
make | tee net.out
