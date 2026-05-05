set -ex

export RUN=49_tiliqua_2layer_8d_interp
export DRD=datalogger_firmware/data/2d_embed_interp/wide_freq_range/24kHz
export FILTER_D=8
export N_INT=4
export N_FRAC=12

[ ! -d runs/$RUN ] && mkdir runs/$RUN

time uv run -m qkeras_version.train \
 --run $RUN \
 --data-root-dir $DRD \
 --num-layers 2 --in-out-d 4 --filter-size $FILTER_D \
 --num-train-egs 20000 --epochs 10 --learning-rate 1e-3 --l2 0.0001 \
 | tee runs/$RUN/qkeras_version.train.out

time uv run -m fxpmath_version.test \
 --data-root-dir $DRD \
 --load-weights runs/$RUN/weights/qkeras/latest.pkl \
 --layer-info runs/$RUN/qkeras_model.layer_info.json \
 --test-x-dir runs/$RUN/test_x_files/ \
 --plot-dir runs/$RUN/ \
 --num-test-egs 300 \
 | tee runs/$RUN/fxpmath_version.test.out

python -m unittest discover test_equivalences

# build & flash
pushd /home/mat/dev/tiliqua/gateware
rm -rf build/neural-waveshaper-r3/
pdm neural_waveshaper build --hw r3 --fs-192khz
grep -A30 ^Info:\ Devi build/neural-waveshaper-r3/top.tim
openFPGALoader -c dirtyJtag build/neural-waveshaper-r3/top.bit
popd