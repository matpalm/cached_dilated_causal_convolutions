SIM ?= icarus
TOPLEVEL_LANG ?= verilog
VERILOG_SOURCES = ../../src/rolling_average.sv
TOPLEVEL = rolling_average
MODULE = rolling_average_tb

COMPILE_ARGS += -Prolling_average.LEN=4

include $(shell cocotb-config --makefiles)/Makefile.sim
