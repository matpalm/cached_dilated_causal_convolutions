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
        print("i", i, "cached", dut.out_l0.value, dut.out_l1.value,
                                dut.out_l2.value, dut.out_l3.value)

    assert dut.out_l0.value == 0x006A  # 0000 0000 0110 1010
    assert dut.out_l1.value == 0x006E  # 0000 0000 0110 1110
    assert dut.out_l2.value == 0x0072  # 0000 0000 0111 0010
    assert dut.out_l3.value == 0x0076  # 0000 0000 0111 0110
