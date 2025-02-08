set -ex

export RUN=39_qb_16d
export DRD=datalogger_firmware/data/2d_embed_interp/wide_freq_range/24kHz
export FILTER_D=16

# [ ! -d runs/$RUN ] && mkdir runs/$RUN

# unset CUDA_VISIBLE_DEVICES
# time python3 -m qkeras_version.train \
#  --run $RUN \
#  --data-root-dir $DRD \
#  --num-layers 3 --in-out-d 4 --filter-size $FILTER_D \
#  --num-train-egs 20000 --epochs 5 --learning-rate 1e-3 --l2 0.0001 \
#  | tee runs/$RUN/qkeras_version.train.out

export CUDA_VISIBLE_DEVICES=""
time python3 -m qkeras_version.test \
 --wave square \
 --data-root-dir $DRD \
 --num-layers 3 \
 --filter-size $FILTER_D \
 --load-weights runs/$RUN/weights/keras/010-0.01833 \
 --test-seq-len 500
 unset CUDA_VISIBLE_DEVICES

# pushd sverilog_version/src
# [ -f network.sv ] && rm network.sv
# ln -s qb_network.sv network.sv
# popd

# # note: make files use FILTER_D
# WAVE=sine ./run_make_network.sh
# WAVE=ramp ./run_make_network.sh
# WAVE=square ./run_make_network.sh
# WAVE=zigzag ./run_make_network.sh
