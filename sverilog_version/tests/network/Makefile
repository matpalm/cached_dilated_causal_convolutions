SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = ../../src/network.sv
VERILOG_SOURCES += ../../src/conv1d.sv
VERILOG_SOURCES += ../../src/row_by_matrix_multiply.sv
VERILOG_SOURCES += ../../src/dot_product.sv
VERILOG_SOURCES += ../../src/left_shift_buffer.sv
VERILOG_SOURCES += ../../src/activation_cache.sv
TOPLEVEL = network
MODULE = network_tb

include $(shell cocotb-config --makefiles)/Makefile.sim
