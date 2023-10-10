import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

# add .. to path so we can import a common test 'util'
#import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#from util import *

# ported from https://github.com/apfelaudio/eurorack-pmod/blob/master/gateware/sim/vca/tb_vca.py

@cocotb.test()
async def test_networks(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    # slurp entries from test_x values
    test_x_hex_values = []
    with open('test_x.hex', 'r') as f:
        for line in f.readlines():
            _i, x, fp_x, fp_x_hex = line.strip().split(" ")
            test_x_hex_values.append((x, fp_x, fp_x_hex))

    # set initial value
    next_x = test_x_hex_values[0][0]
    next_fp_x_hex = test_x_hex_values[0][2]
    test_x_hex_values.pop(0)
    dut.inp.value = eval(next_fp_x_hex)

    def net_state_to_str():
        if dut.net_state.value == 0:
            return "CLK_LSB"
        elif dut.net_state.value == 1:
            return "RST_CONV_0"
        elif dut.net_state.value == 2:
            return "CONV_0_RUNNING"
        elif dut.net_state.value == 3:
            return "CLK_ACT_CACHE_0"
        elif dut.net_state.value == 4:
            return "RST_CONV_1"
        elif dut.net_state.value == 5:
            return "CONV_1_RUNNING"
        elif dut.net_state.value == 6:
            return "CLK_ACT_CACHE_1"
        elif dut.net_state.value == 7:
            return "RST_CONV_2"
        elif dut.net_state.value == 8:
            return "CONV_2_RUNNING"
        elif dut.net_state.value == 9:
            return "OUTPUT"
        else:
            raise Exception(f"unknown state [{dut.net_state.value}]")

    for i in range(100000):

        if dut.net_state.value == 9:
            if len(test_x_hex_values) == 0:
                # we are done
                break
            # prep next test x value
            next_x = test_x_hex_values[0][0]
            next_fp_x_hex = test_x_hex_values[0][2]
            test_x_hex_values.pop(0)
            dut.inp.value = eval(next_fp_x_hex)
            print("|test_x_hex_values|=", len(test_x_hex_values))

        print("i", i, "state", net_state_to_str(), dut.net_state.value)
        print("next_x", next_x, next_fp_x_hex)
        print("lsb", dut.lsb.out.value)
        print("c0_out_v", dut.c0_out_v.value)
        print("conv0.result.value", dut.conv0.result.value)
        print("ac_c0_clk", dut.ac_c0_clk.value)
        print("ac_c0_0 buffer", dut.activation_cache_c0_0.buffer.value)
        print("ac_c0_1 buffer", dut.activation_cache_c0_1.buffer.value)
        print("ac_c0_2 buffer", dut.activation_cache_c0_2.buffer.value)
        print("ac_c0_3 buffer", dut.activation_cache_c0_3.buffer.value)
        print("ac_c0_0 out", dut.activation_cache_c0_0.out.value)
        print("ac_c0_1 out", dut.activation_cache_c0_1.out.value)
        print("ac_c0_2 out", dut.activation_cache_c0_2.out.value)
        print("ac_c0_3 out", dut.activation_cache_c0_3.out.value)
        print("c1_out_v", dut.c1_out_v.value)
        print("conv1.result.value", dut.conv1.result.value)
        print("ac_c1_clk", dut.ac_c1_clk.value)
        print("ac_c1_0 buffer", dut.activation_cache_c1_0.buffer.value)
        print("ac_c1_1 buffer", dut.activation_cache_c1_1.buffer.value)
        print("ac_c1_2 buffer", dut.activation_cache_c1_2.buffer.value)
        print("ac_c1_3 buffer", dut.activation_cache_c1_3.buffer.value)
        print("ac_c1_0 out", dut.activation_cache_c1_0.out.value)
        print("ac_c1_1 out", dut.activation_cache_c1_1.out.value)
        print("ac_c1_2 out", dut.activation_cache_c1_2.out.value)
        print("ac_c1_3 out", dut.activation_cache_c1_3.out.value)
        print("c2_out_v", dut.c2_out_v.value)
        print("conv2.result.value", dut.conv2.result.value)
        print("OUT", dut.out_v.value, dut.out.value)
        await RisingEdge(dut.clk)
