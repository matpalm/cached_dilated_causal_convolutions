set -ex

export RUN=88_4_16_16_4
export DRD=datalogger_firmware/data/2d_embed_interp/wide_freq_range/24kHz
export FILTERS="4 16 16 4"

# 4 12 12 4 works

mkdir runs/$RUN || true

# pre train at FP4.12
time uv run -m qkeras_version.train \
 --run $RUN \
 --data-root-dir $DRD \
 --fp-int 4 --fp-frac 12 \
 --in-out-d 4 --filter-sizes $FILTERS \
 --num-train-egs 100 --epochs 1 --learning-rate 1e-3 --l2 0.0001 \
 | tee runs/$RUN/qkeras_version.train.out

# fine tune at FP4.4
# time uv run -m qkeras_version.train \
#  --run ${RUN}_ft \
#  --data-root-dir $DRD \
#  --fp-int 4 --fp-frac 4 \
#  --in-out-d 4 --filter-sizes 4 8 \
#  --init-weights runs/$RUN/weights/keras/MAKE_LATEST \
#  --num-train-egs 20000 --epochs 5 --learning-rate 1e-4 --l2 0.0001 \
#  | tee runs/$RUN/qkeras_version.finetune.out

# time uv run -m fxpmath_version.test \
#  --data-root-dir $DRD \
#  --load-weights runs/$RUN/weights/qkeras/latest.pkl \
#  --layer-info runs/$RUN/qkeras_model.layer_info.json \
#  --test-x-dir runs/$RUN/test_x_files/ \
#  --plot-dir runs/$RUN/ \
#  --num-test-egs 400 \
#  | tee runs/$RUN/fxpmath_version.test.out

# # quite slow, only required for big changes
#rm runs/$RUN/test_x_files/zigzag/test_network.y_pred_fxp.pkl || true
#uv run python -m unittest discover test_equivalences -k test_network

# build & flash
pushd /home/mat/dev/tiliqua/gateware
rm -rf build/neural-waveshaper-r3/
pdm neural_waveshaper build --hw r3 --fs-192khz
grep -A30 ^Info:\ Devi build/neural-waveshaper-r3/top.tim
openFPGALoader -c dirtyJtag build/neural-waveshaper-r3/top.bit
popd

# #pdm flash archive build/neural-waveshaper-r3/neural-waveshaper*.tar.gz --slot 1 --noconfirm
