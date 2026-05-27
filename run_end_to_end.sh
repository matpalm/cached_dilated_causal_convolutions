set -ex

# qkeras 0.9.0 not compatible with keras from in tf 2.16; force legacy package
export TF_USE_LEGACY_KERAS=1

export MIN_NOTE=A2
export MAX_NOTE=A6

export RUN=185_8_16_16_8_8
export FILTERS="8 16 16 8 8"

# smoke config
# export TRAIN_EGS=2
# export PRETRAIN_EPOCHS=1
# export FINETUNE_EPOCHS=1
# export WAVE_CONFIG="--train-interp"

# sanity config
export TRAIN_EGS=100000
export PRETRAIN_EPOCHS=10
export FINETUNE_EPOCHS=10
export WAVE_CONFIG="--train-interp --harsh --soft-clip --double-interp"

# onight config
# export TRAIN_EGS=100000
# export PRETRAIN_EPOCHS=30
# export FINETUNE_EPOCHS=60
# export WAVE_CONFIG="--train-interp --harsh --soft-clip --double-interp"

# pre train at FP3.15 ( relu4 )
mkdir -p runs/$RUN/pretrain || true
time uv run --with "tensorflow[and-cuda]==2.16.2" --with "tf_keras" -m qkeras_version.train \
 --run $RUN/pretrain \
 --min-note $MIN_NOTE --max-note $MAX_NOTE \
 $WAVE_CONFIG \
 --fp-int 3 --fp-frac 15 \
 --filter-sizes $FILTERS --relu-upper-bound 4 \
 --alpha-mse 1.0 --beta-stft 0.01 --beta-stft-warmup 0.25 --beta-stft-ramp 0.25 \
 --num-train-egs $TRAIN_EGS --epochs $PRETRAIN_EPOCHS --batch-size 64 --learning-rate 1e-3 --l2 1e-4 \
 | tee runs/$RUN/pretrain/qkeras_version.train.out

# time uv run --with "tensorflow[and-cuda]==2.16.2" --with "tf_keras" -m qkeras_version.test \
#  --fp-int 4 --fp-frac 12 \
#  --filter-sizes $FILTERS \
#  --load-weights runs/$RUN/pretrain/weights/keras/ \
#  --wave sine --min-note A4 --max-note A4 \
#  --test-seq-len 1000 \
#  | tee runs/$RUN/pretrain/qkeras_version.test.out

# fine tune at FP3.6 ( relu4 )
mkdir -p runs/$RUN/finetune || true
time uv run --with "tensorflow[and-cuda]==2.16.2" --with "tf_keras" -m qkeras_version.train \
 --run ${RUN}/finetune \
 --min-note $MIN_NOTE --max-note $MAX_NOTE\
 $WAVE_CONFIG \
 --fp-int 3 --fp-frac 6 \
 --filter-sizes $FILTERS --relu-upper-bound 4 \
 --init-weights runs/$RUN/pretrain/weights/keras/ \
 --alpha-mse 1.0 --beta-stft 0.0 \
 --num-train-egs $TRAIN_EGS --epochs $FINETUNE_EPOCHS --batch-size 64 --learning-rate 1e-4 --l2 1e-4 \
 | tee runs/$RUN/finetune/qkeras_version.train.out

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

# build both versions
# what a load of hack o_O
export RUN_DIR=$PWD/runs/$RUN/
pdm_build() {
    pushd /home/mat/dev/tiliqua/gateware
    export N_INT=$1
    export N_FRAC=$2
    export SUB_RUN=$3
    export WEIGHTS_PKL=$RUN_DIR/$SUB_RUN/weights/qkeras/latest.pkl
    time pdm neural_waveshaper build --hw r3 --fs-192khz --name "nw_${RUN}_${SUB_RUN}"
    popd
    cp -r /home/mat/dev/tiliqua/gateware/build/nw_${RUN}_${SUB_RUN}* runs/$RUN/$SUB_RUN/neural-waveshaper-r3
    uv run -m amaranth_version.parse_top_tim \
      --top-tim runs/$RUN/$SUB_RUN/neural-waveshaper-r3 \
      | tee runs/$RUN/$SUB_RUN/parsed_top_tim
}

#pdm_build 3 15 pretrain &
pdm_build 3 6 finetune &
wait

#openFPGALoader -c dirtyJtag build/neural-waveshaper-r3/top.bit || true
#popd
#cp -r /home/mat/dev/tiliqua/gateware/build/neural-waveshaper-r3/ runs/$RUN/$SUB_RUN
#uv run -m amaranth_version.parse_top_tim --top-tim runs/$RUN/$SUB_RUN/neural-waveshaper-r3/top.tim

# #pdm flash archive build/neural-waveshaper-r3/neural-waveshaper*.tar.gz --slot 1 --noconfirm
