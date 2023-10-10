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
async def test_conv1d(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    # first pass without relu

    dut.apply_relu.value = 0

    dut.a0_d0.value = 0x0400   # 0000.010000000000 0.25
    dut.a0_d1.value = 0xFDFC   # 1111.110111111100 -0.1259765625
    dut.a0_d2.value = 0x0506   # 0000.010100000110 0.31396484375
    dut.a0_d3.value = 0xF000   # 1111.000000000000 -1.0

    dut.a1_d0.value = 0x0400   # 0000.010000000000 0.25
    dut.a1_d1.value = 0xFDFC   # 1111.110111111100 -0.1259765625
    dut.a1_d2.value = 0x0506   # 0000.010100000110 0.31396484375
    dut.a1_d3.value = 0xF000   # 1111.000000000000 -1.0

    dut.a2_d0.value = 0x0400   # 0000.010000000000 0.25
    dut.a2_d1.value = 0xFDFC   # 1111.110111111100 -0.1259765625
    dut.a2_d2.value = 0x0506   # 0000.010100000110 0.31396484375
    dut.a2_d3.value = 0xF000   # 1111.000000000000 -1.0

    dut.a3_d0.value = 0x0400   # 0000.010000000000 0.25
    dut.a3_d1.value = 0xFDFC   # 1111.110111111100 -0.1259765625
    dut.a3_d2.value = 0x0506   # 0000.010100000110 0.31396484375
    dut.a3_d3.value = 0xF000   # 1111.000000000000 -1.0

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    for i in range(10):
        print("i", i, "c1d_state", dut.c1d_state.value)
        print("accum      ", dut.accum.value)
        print("result     ", dut.result.value)
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    assert dut.out_v.value == 1
    assert dut.out_d0.value == 0xFE7E  # 1111 1110 0111 1110
    assert dut.out_d1.value == 0xF716  # 1111 0111 0001 0110
    assert dut.out_d2.value == 0x0326  # 0000 0011 0010 0110
    assert dut.out_d3.value == 0x0E61  # 0000 1110 0110 0001

    # same again with RELU

    dut.apply_relu.value = 1

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    for i in range(10):
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    assert dut.out_v.value == 1

    assert dut.out_d0.value == 0
    assert dut.out_d1.value == 0
    assert dut.out_d2.value == 0x0326  # 0000 0011 0010 0110
    assert dut.out_d3.value == 0x0E61  # 0000 1110 0110 0001