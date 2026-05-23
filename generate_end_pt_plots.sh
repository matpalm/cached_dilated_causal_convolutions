uv run -m tf_data_pipeline.quadrature_data --output-png endpts.hF.scF.png &
uv run -m tf_data_pipeline.quadrature_data --output-png endpts.hT.scF.png --harsh &
uv run -m tf_data_pipeline.quadrature_data --output-png endpts.hF.scT.png --soft-clip &
uv run -m tf_data_pipeline.quadrature_data --output-png endpts.hT.scT.png --harsh --soft-clip &
wait
