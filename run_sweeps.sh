uv run -m qkeras_version.test --tri-freq 300 --run $1/pretrain/ --test-set a_cv
uv run -m qkeras_version.test --tri-freq 300 --run $1/pretrain/ --test-set b_cv
uv run -m qkeras_version.test --tri-freq 300 --run $1/pretrain/ --test-set morph_sine
uv run -m qkeras_version.test --tri-freq 300 --run $1/pretrain/ --test-set morph_ramp
uv run -m qkeras_version.test --tri-freq 300 --run $1/pretrain/ --test-set morph_sqr_ramp
uv run -m qkeras_version.test --tri-freq 300 --run $1/pretrain/ --test-set symmetry
