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
async def test_left_shift_buffer(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    for i in range(10):
        dut.inp.value = i
        await RisingEdge(dut.clk)
        print("i", dut.out_d0.value, dut.out_d1.value,
                   dut.out_d2.value, dut.out_d3.value)

    assert dut.out_d0.value == 5
    assert dut.out_d1.value == 6
    assert dut.out_d2.value == 7
    assert dut.out_d3.value == 8
