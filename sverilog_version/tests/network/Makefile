SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = ../../src/network.sv
VERILOG_SOURCES += ../../src/activation_cache.sv
VERILOG_SOURCES += ../../src/conv1d.sv
VERILOG_SOURCES += ../../src/dot_product.sv
VERILOG_SOURCES += ../../src/left_shift_buffer.sv
VERILOG_SOURCES += ../../src/row_by_matrix_multiply.sv
VERILOG_SOURCES += ../../src/po2_conv1d.sv
VERILOG_SOURCES += ../../src/po2_dot_product.sv
VERILOG_SOURCES += ../../src/po2_multiply.sv
VERILOG_SOURCES += ../../src/po2_row_by_matrix_multiply.sv
TOPLEVEL = network
MODULE = network_tb

COMPILE_ARGS += -Pnetwork.FILTER_D=${FILTER_D}
COMPILE_ARGS += -Pnetwork.FILTER_PO2_D=${FILTER_PO2_D}

include $(shell cocotb-config --makefiles)/Makefile.sim
