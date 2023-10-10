import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

# add .. to path so we can import a common test 'util'
#import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#from util import *

@cocotb.test()
async def test_activation_cache(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    for i in range(120):
        dut.inp.value = i
        await RisingEdge(dut.clk)
        print("i", i, "cached", dut.out.value)

    assert dut.out.value == [70, 86, 102, 118]
