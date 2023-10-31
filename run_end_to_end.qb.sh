set -ex

export RUN=27_qb__layer_regression_test
export DRD=datalogger_firmware/data/2d_embed_interp/wide_freq_range/24kHz
export FILTER_D=16


[ ! -d runs/$RUN ] && mkdir runs/$RUN

unset CUDA_VISIBLE_DEVICES
time python3 -m qkeras_version.train \
 --run $RUN \
 --data-root-dir $DRD \
 --num-layers 3 --in-out-d 4 --filter-size $FILTER_D \
 --num-train-egs 20000 --epochs 5 --learning-rate 1e-3 --l2 0.0001 \
 | tee runs/$RUN/qkeras_version.train.out

export CUDA_VISIBLE_DEVICES=""
time python3 -m fxpmath_version.test \
 --data-root-dir $DRD \
 --load-weights runs/$RUN/weights/qkeras/latest.pkl \
 --layer-info runs/$RUN/qkeras_model.layer_info.json \
 --test-x-dir runs/$RUN/test_x_files/ \
 --plot-dir runs/$RUN/ \
 --write-verilog-weights runs/$RUN/weights/verilog/latest \
 --num-test-egs 200 \
 | tee runs/$RUN/fxpmath_version.test.out
unset CUDA_VISIBLE_DEVICES

# note: make files use FILTER_D
WAVE=sine ./run_make_network.sh
WAVE=ramp ./run_make_network.sh
WAVE=square ./run_make_network.sh
WAVE=zigzag ./run_make_network.sh
