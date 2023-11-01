SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = ../../src/po2_row_by_matrix_multiply.sv
VERILOG_SOURCES += ../../src/po2_dot_product.sv
VERILOG_SOURCES += ../../src/po2_multiply.sv
TOPLEVEL = po2_row_by_matrix_multiply
MODULE = po2_row_by_matrix_multiply_tb

COMPILE_ARGS += -Ppo2_row_by_matrix_multiply.IN_D=8
COMPILE_ARGS += -Ppo2_row_by_matrix_multiply.OUT_D=4
COMPILE_ARGS += -Ppo2_row_by_matrix_multiply.WEIGHTS=\"test_matrix_8_4\"

include $(shell cocotb-config --makefiles)/Makefile.sim
