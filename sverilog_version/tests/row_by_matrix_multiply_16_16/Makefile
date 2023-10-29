SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = ../../src/row_by_matrix_multiply.sv
VERILOG_SOURCES += ../../src/dot_product.sv
TOPLEVEL = row_by_matrix_multiply
MODULE = row_by_matrix_multiply_tb

COMPILE_ARGS += -Prow_by_matrix_multiply.IN_D=16
COMPILE_ARGS += -Prow_by_matrix_multiply.OUT_D=16
COMPILE_ARGS += -Prow_by_matrix_multiply.WEIGHTS=\"test_matrix_16_16\"

include $(shell cocotb-config --makefiles)/Makefile.sim
