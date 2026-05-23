set -ex

# qkeras 0.9.0 not compatible with keras from in tf 2.16; force legacy package
export TF_USE_LEGACY_KERAS=1

export MIN_NOTE=A4
export MAX_NOTE=A4

export RUN=152_4_8_16_16_relu4_fp3_6
export FILTERS="4 8 16 16"

mkdir -p runs/$RUN/{pretrain,fine_tune} || true

# pre train at FP3.15 ( relu4 )
time uv run --with "tensorflow[and-cuda]==2.16.2" --with "tf_keras" -m qkeras_version.train \
 --run $RUN/pretrain \
 --min-note $MIN_NOTE --max-note $MAX_NOTE --train-interp --soft-clip \
 --fp-int 3 --fp-frac 15 \
 --filter-sizes $FILTERS --relu-upper-bound 4 \
 --alpha-mse 1.0 --beta-stft 0.01 --beta-stft-ramp-epochs 10 \
 --num-train-egs 1000 --epochs 1 --learning-rate 1e-3 --l2 0.0001 \
 | tee runs/$RUN/pretrain/qkeras_version.train.out

# --harsh-waves --soft-clip \

# time uv run --with "tensorflow[and-cuda]==2.16.2" --with "tf_keras" -m qkeras_version.test \
#  --fp-int 4 --fp-frac 12 \
#  --filter-sizes $FILTERS \
#  --load-weights runs/$RUN/pretrain/weights/keras/ \
#  --wave sine --min-note A4 --max-note A4 \
#  --test-seq-len 1000 \
#  | tee runs/$RUN/pretrain/qkeras_version.test.out

# fine tune at FP3.6 ( relu4 )
time uv run --with "tensorflow[and-cuda]==2.16.2" --with "tf_keras" -m qkeras_version.train \
 --run ${RUN}/fine_tune \
 --min-note $MIN_NOTE --max-note $MAX_NOTE --train-interp --soft-clip \
 --fp-int 3 --fp-frac 6 \
 --filter-sizes $FILTERS --relu-upper-bound 4 \
 --init-weights runs/$RUN/pretrain/weights/keras/ \
 --alpha-mse 1.0 --beta-stft 0.01 --beta-stft-ramp-epochs 15 \
 --num-train-egs 1000 --epochs 1 --batch-size 128 --learning-rate 1e-5 --l2 0.0001 \
 | tee runs/$RUN/fine_tune/qkeras_version.train.out

# time uv run -m fxpmath_version.test \
#  --min-note A4 --max-note A4 \
#  --load-weights runs/$RUN/??/weights/qkeras/latest.pkl \
#  --layer-info runs/$RUN/??/qkeras_model.layer_info.json \
#  --test-x-dir runs/$RUN/??/test_x_files/ \
#  --plot-dir runs/$RUN/ \
#  --num-test-egs 1000 \
#  | tee runs/$RUN/??/fxpmath_version.test.out

# # quite slow, only required for big changes
# rm runs/$RUN/test_x_files/zigzag/test_network.y_pred_fxp.pkl || true
# uv run python -m unittest discover test_equivalences -k test_network

# build & flash
export N_INT=3
export N_FRAC=6
export WEIGHTS_PKL=runs/${RUN}/fine_tune/weights/qkeras/latest.pkl
pushd /home/mat/dev/tiliqua/gateware
rm -rf build/neural-waveshaper-r3/
time pdm neural_waveshaper build --hw r3 --fs-192khz
grep -A30 ^Info:\ Devi build/neural-waveshaper-r3/top.tim
openFPGALoader -c dirtyJtag build/neural-waveshaper-r3/top.bit || true
popd
cp -r /home/mat/dev/tiliqua/gateware/build/neural-waveshaper-r3/ runs/$RUN

# #pdm flash archive build/neural-waveshaper-r3/neural-waveshaper*.tar.gz --slot 1 --noconfirm
