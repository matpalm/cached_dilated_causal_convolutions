#!/usr/bin/env bash
set -ex
WAVE=sine ./run_plot_network_output.sh
WAVE=ramp ./run_plot_network_output.sh
WAVE=square ./run_plot_network_output.sh
WAVE=zigzag ./run_plot_network_output.sh
