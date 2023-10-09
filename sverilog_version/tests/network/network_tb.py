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

    # first pass without relu

    dut.inp.value = 0x1000

    for i in range(150):
        print("i", i, "state", dut.net_state)
        print("lsb", dut.lsb.out.value)
        print("c0_out_v", dut.c0_out_v.value, "dut.conv0.result.value", dut.conv0.result.value)
        print("ac_c0_clk", dut.ac_c0_clk.value)
        print("ac_c0_0 buffer", dut.activation_cache_c0_0.buffer.value)
        print("ac_c0_1 buffer", dut.activation_cache_c0_1.buffer.value)
        print("ac_c0_2 buffer", dut.activation_cache_c0_2.buffer.value)
        print("ac_c0_3 buffer", dut.activation_cache_c0_3.buffer.value)
        print("OUT", dut.out_v.value, dut.out.value)
        await RisingEdge(dut.clk)
