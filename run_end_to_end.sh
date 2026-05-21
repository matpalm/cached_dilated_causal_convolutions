set -ex

# qkeras 0.9.0 not compatible with keras from in tf 2.16; force legacy package
export TF_USE_LEGACY_KERAS=1

export RUN=137_8_16_4_quadrature_harsh_softclip
export FILTERS="8 16 4"

mkdir runs/$RUN || true

# pre train at FP4.12
time uv run --with "tensorflow[and-cuda]==2.16.2" --with "tf_keras" -m qkeras_version.train \
 --run $RUN \
 --min-note A4 --max-note A5 --sample-rate-khz 192 --train-interp \
 --fp-int 4 --fp-frac 12 \
 --in-out-d 4 --filter-sizes $FILTERS \
 --alpha-mse 1.0 --beta-stft 0.1 --beta-stft-ramp-epochs 10 \
 --num-train-egs 20000 --epochs 20 --learning-rate 1e-3 --l2 0.0001 \
 | tee runs/$RUN/qkeras_version.train.out

# time uv run --with "tensorflow[and-cuda]==2.16.2" --with "tf_keras" -m qkeras_version.test \
#  --fp-int 4 --fp-frac 12 \
#  --filter-sizes $FILTERS \
#  --load-weights runs/$RUN/weights/keras/ \
#  --wave sine --min-note A4 --max-note A4 \
#  --test-seq-len 200 \
#  | tee runs/$RUN/qkeras_version.test.out

# fine tune at FP4.4
# time uv run -m qkeras_version.train \
#  --run ${RUN}_ft \
#  --fp-int 4 --fp-frac 4 \
#  --in-out-d 4 --filter-sizes 4 8 \
#  --init-weights runs/$RUN/weights/keras/MAKE_LATEST \
#  --num-train-egs 20000 --epochs 5 --learning-rate 1e-4 --l2 0.0001 \
#  | tee runs/$RUN/qkeras_version.finetune.out

# time uv run -m fxpmath_version.test \
#  --min-note A4 --max-note A4 \
#  --load-weights runs/$RUN/weights/qkeras/latest.pkl \
#  --layer-info runs/$RUN/qkeras_model.layer_info.json \
#  --test-x-dir runs/$RUN/test_x_files/ \
#  --plot-dir runs/$RUN/ \
#  --num-test-egs 50 \
#  | tee runs/$RUN/fxpmath_version.test.out

# # quite slow, only required for big changes
# rm runs/$RUN/test_x_files/zigzag/test_network.y_pred_fxp.pkl || true
# uv run python -m unittest discover test_equivalences -k test_network

# build & flash
# pushd /home/mat/dev/tiliqua/gateware
# rm -rf build/neural-waveshaper-r3/
# pdm neural_waveshaper build --hw r3 --fs-192khz
# grep -A30 ^Info:\ Devi build/neural-waveshaper-r3/top.tim
# openFPGALoader -c dirtyJtag build/neural-waveshaper-r3/top.bit || true
# popd
# cp -r /home/mat/dev/tiliqua/gateware/build/neural-waveshaper-r3/ runs/$RUN

# #pdm flash archive build/neural-waveshaper-r3/neural-waveshaper*.tar.gz --slot 1 --noconfirm
