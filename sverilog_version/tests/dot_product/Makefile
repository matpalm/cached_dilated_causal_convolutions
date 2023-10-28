SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = ../../src/dot_product.sv
TOPLEVEL = dot_product
MODULE = dot_product_tb

COMPILE_ARGS += -Pdot_product.D=16
COMPILE_ARGS += -Pdot_product.B_VALUES=\"test_b_values.hex\"

include $(shell cocotb-config --makefiles)/Makefile.sim
