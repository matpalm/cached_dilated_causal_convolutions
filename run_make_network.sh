#!/usr/bin/env bash
set -ex
cd ~/dev/cached_dilated_causal_convolutions/sverilog_version/tests/network
make | tee net.out
