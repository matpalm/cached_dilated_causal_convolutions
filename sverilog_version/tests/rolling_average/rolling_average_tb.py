import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

# add .. to path so we can import a common test 'util'
#import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#from util import *

def hex_if_not_x(s):
    try:
        return hex(s)
    except ValueError:
        # probably xxxxxxxxxx
        return s

@cocotb.test()
async def test_activation_cache(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    for i in range(5):
        print(i, dut.out.value)
        dut.inp.value = 0
        await RisingEdge(dut.clk)
    assert dut.out.value == 0

    for i in range(10):
        print(i, dut.out.value)
        dut.inp.value = 10
        await RisingEdge(dut.clk)
    assert dut.out.value == 10

    for i in range(2):
        print(i, dut.out.value)
        dut.inp.value = 0
        await RisingEdge(dut.clk)
    assert dut.out.value == 2

    #                             hex_if_not_x(dut.out_l2.value), hex_if_not_x(dut.out_l3.value))

    # assert dut.out_l0.value == 0x006A_006C  # 0000 0000 0110 1010   & it's +2 value
    # assert dut.out_l1.value == 0x006E_0070  # 0000 0000 0110 1110
    # assert dut.out_l2.value == 0x0072_0074  # 0000 0000 0111 0010
    # assert dut.out_l3.value == 0x0076_0078  # 0000 0000 0111 0110
