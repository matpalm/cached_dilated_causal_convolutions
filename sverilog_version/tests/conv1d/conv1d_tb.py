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

    dut.packed_a0.value = 0x0400_FDFC_0506_F000_0400_FDFC_0506_F000
    dut.packed_a1.value = 0x0400_FDFC_0506_F000_0400_FDFC_0506_F000
    dut.packed_a2.value = 0x0400_FDFC_0506_F000_0400_FDFC_0506_F000
    dut.packed_a3.value = 0x0400_FDFC_0506_F000_0400_FDFC_0506_F000

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    for i in range(30):
        print("i", i, "c1d_state", dut.c1d_state.value)
        print("accum      ", dut.accum.value)
        print("result     ", dut.result.value)
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    assert dut.out_v.value == 1
    assert dut.packed_out.value == 0x16e5_febf_1642_f8e9_0ba3_1ca9_076e_11c3

    # same again with RELU

    dut.apply_relu.value = 1

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    for i in range(30):
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    assert dut.out_v.value == 1
    assert dut.packed_out.value == 0x16e5_0000_1642_0000_0ba3_1ca9_076e_11c3
