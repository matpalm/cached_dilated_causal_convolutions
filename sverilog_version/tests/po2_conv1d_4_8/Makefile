SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = ../../src/po2_conv1d.sv
VERILOG_SOURCES += ../../src/po2_row_by_matrix_multiply.sv
VERILOG_SOURCES += ../../src/po2_dot_product.sv
VERILOG_SOURCES += ../../src/po2_multiply.sv
TOPLEVEL = po2_conv1d
MODULE = po2_conv1d_tb

COMPILE_ARGS += -Ppo2_conv1d.IN_D=4
COMPILE_ARGS += -Ppo2_conv1d.OUT_D=8
COMPILE_ARGS += -Ppo2_conv1d.WEIGHTS=\"test_weights_4_8\"

include $(shell cocotb-config --makefiles)/Makefile.sim
