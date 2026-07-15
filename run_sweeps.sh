# uv run -m qkeras_version.test --tri-freq 300 --run $1/$2/ --test-set a_cv
# uv run -m qkeras_version.test --tri-freq 300 --run $1/$2/ --test-set b_cv
# uv run -m qkeras_version.test --tri-freq 300 --run $1/$2/ --test-set morph_sine
# uv run -m qkeras_version.test --tri-freq 300 --run $1/$2/ --test-set morph_ramp
# uv run -m qkeras_version.test --tri-freq 300 --run $1/$2/ --test-set morph_sqr_ramp
# uv run -m qkeras_version.test --tri-freq 300 --run $1/$2/ --test-set symmetry

nice uv run python -m amaranth_version.tests.simulate \
 --run $1 --sub-run $2 \
 --a-cv -0.6 --b-cv -0.6 --morph-cv -1 --output-plot square.jpg &

nice uv run python -m amaranth_version.tests.simulate \
 --run $1 --sub-run $2 \
 --a-cv -0.2 --b-cv -0.2 --morph-cv -1 --output-plot ramp.jpg &

nice uv run python -m amaranth_version.tests.simulate \
 --run $1 --sub-run $2 \
 --a-cv 0.2 --b-cv 0.2 --morph-cv -1 --output-plot triangle.jpg &

nice uv run python -m amaranth_version.tests.simulate \
 --run $1 --sub-run $2 \
 --a-cv 0.6 --b-cv 0.6 --morph-cv -1 --output-plot sine.jpg &

wait