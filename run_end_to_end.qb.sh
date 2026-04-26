set -ex

export RUN=35_qb_FP4_12
export DRD=datalogger_firmware/data/2d_embed_interp/wide_freq_range/24kHz
export FILTER_D=4

[ ! -d runs/$RUN ] && mkdir runs/$RUN

time uv run -m qkeras_version.train \
 --run $RUN \
 --data-root-dir $DRD \
 --n-int 4 --n-frac 12 \
 --num-layers 3 --in-out-d 4 --filter-size $FILTER_D \
 --num-train-egs 20000 --epochs 10 --learning-rate 1e-3 --l2 0.0001 \
 | tee runs/$RUN/qkeras_version.train.out

time uv run -m fxpmath_version.test \
 --data-root-dir $DRD \
 --load-weights runs/$RUN/weights/qkeras/latest.pkl \
 --layer-info runs/$RUN/qkeras_model.layer_info.json \
 --test-x-dir runs/$RUN/test_x_files/ \
 --plot-dir runs/$RUN/ \
 --write-verilog-weights runs/$RUN/weights/verilog/latest \
 --num-test-egs 500 \
 | tee runs/$RUN/fxpmath_version.test.out

echo "VERILOG VERSION DOESNT WORK IN EXISTING UV ENV"

# pushd sverilog_version/src
# [ -f network.sv ] && rm network.sv
# ln -s qb_network.sv network.sv
# popd

# # note: make files use FILTER_D
# WAVE=sine ./run_make_network.sh
# WAVE=ramp ./run_make_network.sh
# WAVE=square ./run_make_network.sh
# WAVE=zigzag ./run_make_network.sh
