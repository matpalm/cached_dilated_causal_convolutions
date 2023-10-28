SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = ../../src/activation_cache.sv
TOPLEVEL = activation_cache
MODULE = activation_cache_tb

COMPILE_ARGS += -Pactivation_cache.D=2

include $(shell cocotb-config --makefiles)/Makefile.sim
