set -ex

# qkeras 0.9.0 not compatible with keras from in tf 2.16; force legacy package
export TF_USE_LEGACY_KERAS=1

#export RUN=229_test_no_quantise_output
export RUN=234_pc600

export FILTERS="16 16 16 16"

export RUN_ID=`echo $RUN | cut -d'_' -f1`
export PSRAM_ACTIVATION_CACHE_INDICES="[-1]"
export BUILD=nw_${RUN_ID}_psram-1

# smoke config
# export TRAIN_EGS=1_000
# export PRETRAIN_EPOCHS=1
# export FINETUNE_EPOCHS=1

# sanity config
# export TRAIN_EGS=10_000
# export PRETRAIN_EPOCHS=20
# export FINETUNE_EPOCHS=10

# onight config
export TRAIN_EGS=100_000
export PRETRAIN_EPOCHS=30
export FINETUNE_EPOCHS=60

export SAMPLE_RATE_KHZ=48
export BATCH_SIZE=32

# --quantise-output
pretrain() {
    # pre train at FP3.15 ( relu4 )
    mkdir -p runs/$RUN/pretrain || true
    time uv run -m qkeras_version.train_pc \
        --run $RUN/pretrain \
        --sample-rate-khz $SAMPLE_RATE_KHZ \
        --train-seq-len-multiplier 2 \
        --capture-run 600 --keras-model 232_keras/i9 \
        --fp-int 3 --fp-frac 15 \
        --filter-sizes $FILTERS --relu-upper-bound 4 \
        --alpha-mse 1.0 --use-huber-loss --beta-stft 0.01 --beta-stft-warmup 0.25 --beta-stft-ramp 0.25 \
        --num-train-egs $TRAIN_EGS --epochs $PRETRAIN_EPOCHS --batch-size $BATCH_SIZE \
        --learning-rate 1e-3 --l2 1e-4 \
        | tee runs/$RUN/pretrain/qkeras_version.train.out
}

finetune() {
    # fine tune at FP3.6 ( relu4 )
    mkdir -p runs/$RUN/finetune || true
    time uv run -m qkeras_version.train_pc \
        --run ${RUN}/finetune \
        --sample-rate-khz $SAMPLE_RATE_KHZ \
        --train-seq-len-multiplier 2 \
        --capture-run 600 --keras-model 232_keras/i9 \
        --fp-int 3 --fp-frac 6 \
        --filter-sizes $FILTERS --relu-upper-bound 4 \
        --init-weights runs/$RUN/pretrain/weights/keras/ \
        --alpha-mse 1.0 --use-huber-loss --beta-stft 0.01 --beta-stft-warmup 0.75 --beta-stft-ramp 0.25 \
        --num-train-egs $TRAIN_EGS --epochs $FINETUNE_EPOCHS --batch-size $BATCH_SIZE \
        --learning-rate 1e-4 --l2 1e-4 \
        | tee runs/$RUN/finetune/qkeras_version.train.out
}

fxp_math_equiv_test() {
    export SUB_RUN=$1
    # quite slow, only required for big changes ( and depend on fxpmath_version.test )
    # rm -rf runs/$RUN/$SUB_RUN/test_x_files/ || true
    # time uv run -m fxpmath_version.test \
    # --min-note A4 --max-note A4 \
    # --load-weights runs/$RUN/$SUB_RUN/weights/qkeras/latest.pkl \
    # --layer-info runs/$RUN/$SUB_RUN/qkeras_model.layer_info.json \
    # --wave sine \
    # --test-x-dir runs/$RUN/$SUB_RUN/test_x_files/ \
    # --plot-dir runs/$RUN/ \
    # --num-test-egs 50 \
    # | tee runs/$RUN/$SUB_RUN/fxpmath_version.test.out
    # rm runs/$RUN/$SUB_RUN/test_x_files/sine/test_network.y_pred_fxp.pkl || true
    uv run python -m unittest discover test_equivalences -k test_network
}

pretrain
finetune
#fxp_math_equiv_test finetune

# build both versions
# what a load of hack o_O
export RUN_DIR=$PWD/runs/$RUN/
export HW=r3
pdm_build() {
    pushd /home/mat/dev/tiliqua/gateware
    export N_INT=$1
    export N_FRAC=$2
    export SUB_RUN=$3
    export WEIGHTS_PKL=$RUN_DIR/$SUB_RUN/weights/qkeras/latest.pkl
    time pdm neural_waveshaper build --hw $HW --name $BUILD --fs-192khz
    popd
    cp -r /home/mat/dev/tiliqua/gateware/build/$BUILD-${HW} runs/$RUN/$SUB_RUN/
    uv run -m amaranth_version.parse_top_tim \
      --top-tim runs/$RUN/$SUB_RUN/$BUILD-${HW} \
      | tee runs/$RUN/$SUB_RUN/parsed_top_tim
}

#pdm_build 3 15 pretrain &
#pdm_build 3 6 finetune &
wait

#openFPGALoader -c dirtyJtag build/neural-waveshaper-r3/top.bit || true
#popd
#cp -r /home/mat/dev/tiliqua/gateware/build/neural-waveshaper-r3/ runs/$RUN/$SUB_RUN
#uv run -m amaranth_version.parse_top_tim --top-tim runs/$RUN/$SUB_RUN/neural-waveshaper-r3/top.tim

# #pdm flash archive build/neural-waveshaper-r3/neural-waveshaper*.tar.gz --slot 1 --noconfirm
