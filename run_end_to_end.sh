set -ex

# qkeras 0.9.0 not compatible with keras from in tf 2.16; force legacy package
export TF_USE_LEGACY_KERAS=1

#export RUN=261_pc600_candidate
#export RUN=262_pc600_rnd_flip
#export RUN=263_pc600_rnd_flip_no_skip
#export RUN=264_pc600_smaller
#export RUN=265_pc600_smaller  # 8_8_8
export RUN=266_pc600_16x3
#export RUN=267_pc600_16x4
#export RUN=268_pc600_16x5
#export RUN=269_pc600_24x4_ps1

export FILTERS="16 16 16"

export RUN_ID=`echo $RUN | cut -d'_' -f1`
export PSRAM_ACTIVATION_CACHE_INDICES="[-1]"
export BUILD=nw_${RUN_ID}_ps1
export RUN_DIR=$PWD/runs/$RUN/
export HW=r3

# smoke config
# export TRAIN_EGS=1_000
# export PRETRAIN_EPOCHS=10
# export FINETUNE_EPOCHS=5

# sanity config
export TRAIN_EGS=20_000
export PRETRAIN_EPOCHS=20
export FINETUNE_EPOCHS=10

# onight config
# export TRAIN_EGS=100_000
# export PRETRAIN_EPOCHS=100
# export FINETUNE_EPOCHS=60

export SAMPLE_RATE_KHZ=48
export BATCH_SIZE=32

# --skip-project-dim 8 \

pretrain() {
    # pre train at FP3.15 ( relu4 )
    mkdir -p runs/$RUN/pretrain || true
    time uv run -m qkeras_version.train \
        --run $RUN/pretrain \
        --sample-rate-khz $SAMPLE_RATE_KHZ \
        --train-seq-len-multiplier 2 \
        --capture-run 600 --keras-model 232_keras/i9 \
        --fp-int 3 --fp-frac 15 \
        --filter-sizes $FILTERS --relu-upper-bound 4 \
        --alpha-mse 1.0 --use-huber-loss --beta-stft 0.01 --beta-stft-warmup 5 --beta-stft-ramp 5 \
        --num-train-egs $TRAIN_EGS --epochs $PRETRAIN_EPOCHS --batch-size $BATCH_SIZE \
        --learning-rate 1e-3 --l2 1e-4 \
        | tee runs/$RUN/pretrain/qkeras_version.train.out
}

finetune() {
    export N_INT=$1
    export N_FRAC=$2
    mkdir -p runs/$RUN/finetune_${N_INT}_${N_FRAC} || true
    time uv run -m qkeras_version.train \
        --run ${RUN}/finetune_${N_INT}_${N_FRAC} \
        --sample-rate-khz $SAMPLE_RATE_KHZ \
        --train-seq-len-multiplier 2 \
        --capture-run 600 --keras-model 232_keras/i9 \
        --fp-int $N_INT --fp-frac $N_FRAC \
        --filter-sizes $FILTERS --relu-upper-bound 4 \
        --init-weights runs/$RUN/pretrain/weights/keras/ \
        --alpha-mse 1.0 --use-huber-loss --beta-stft 0.001 --beta-stft-warmup 5 --beta-stft-ramp 5 \
        --num-train-egs $TRAIN_EGS --epochs $FINETUNE_EPOCHS --batch-size $BATCH_SIZE \
        --learning-rate 1e-4 --l2 1e-4 \
        | tee runs/$RUN/finetune_${N_INT}_${N_FRAC}/qkeras_version.train.out
}

pdm_build() {
    pushd /home/mat/dev/tiliqua/gateware
    export N_INT=$1
    export N_FRAC=$2
    export SUB_RUN=$3
    export WEIGHTS_PKL=$RUN_DIR/$SUB_RUN/weights/qkeras/latest.pkl
    time pdm neural_waveshaper build --hw $HW --name $BUILD # --fs-192khz
    popd
    cp -r /home/mat/dev/tiliqua/gateware/build/$BUILD-${HW} runs/$RUN/$SUB_RUN/
    uv run -m amaranth_version.parse_top_tim \
      --top-tim runs/$RUN/$SUB_RUN/$BUILD-${HW} \
      | tee runs/$RUN/$SUB_RUN/parsed_top_tim
}

#pretrain
#finetune 3 6
finetune 3 8
#pdm_build 3 15 pretrain &   # wont' work with psram activation constraints :/
#pdm_build 3 6 finetune_3_6 &
pdm_build 3 8 finetune_3_8 &
wait

#openFPGALoader -c dirtyJtag build/neural-waveshaper-r3/top.bit || true
#popd
#cp -r /home/mat/dev/tiliqua/gateware/build/neural-waveshaper-r3/ runs/$RUN/$SUB_RUN
#uv run -m amaranth_version.parse_top_tim --top-tim runs/$RUN/$SUB_RUN/neural-waveshaper-r3/top.tim

# #pdm flash archive build/neural-waveshaper-r3/neural-waveshaper*.tar.gz --slot 1 --noconfirm


# fxp_math_equiv_test() {
    # export SUB_RUN=$1
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
    # uv run python -m unittest discover test_equivalences -k test_network
# }