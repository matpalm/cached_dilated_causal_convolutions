#!/usr/bin/env bash
set -ex
WAVE=sine ./run_make_network.sh
WAVE=ramp ./run_make_network.sh
WAVE=square ./run_make_network.sh
WAVE=zigzag ./run_make_network.sh
