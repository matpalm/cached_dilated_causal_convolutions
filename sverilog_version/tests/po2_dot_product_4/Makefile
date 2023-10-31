SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = ../../src/po2_dot_product.sv
VERILOG_SOURCES += ../../src/po2_multiply.sv
TOPLEVEL = po2_dot_product
MODULE = po2_dot_product_tb

COMPILE_ARGS += -Ppo2_dot_product.D=4
COMPILE_ARGS += -Ppo2_dot_product.WEIGHTS=\"test_weights\"

include $(shell cocotb-config --makefiles)/Makefile.sim
