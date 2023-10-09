SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = ../../src/conv1d.sv
VERILOG_SOURCES += ../../src/row_by_matrix_multiply.sv
VERILOG_SOURCES += ../../src/dot_product.sv
TOPLEVEL = conv1d
MODULE = conv1d_tb

include $(shell cocotb-config --makefiles)/Makefile.sim
