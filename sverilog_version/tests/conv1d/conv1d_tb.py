import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

# add .. to path so we can import a common test 'util'
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tb_util import *

@cocotb.test()
async def test_conv1d(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    # first pass without relu

    dut.apply_relu.value = 0

    #                       0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
    # dut.packed_a0.value = 0x0400_FDFC_0506_F000 #_0400_FDFC_0506_F000_0000_0000_0000_0000_0000_0000_0000_0000
    # dut.packed_a1.value = 0x0400_FDFC_0506_F000 #_0400_FDFC_0506_F000_0000_0000_0000_0000_0000_0000_0000_0000
    # dut.packed_a2.value = 0x0400_FDFC_0506_F000 #_0400_FDFC_0506_F000_0000_0000_0000_0000_0000_0000_0000_0000
    # dut.packed_a3.value = 0x0400_FDFC_0506_F000 #_0400_FDFC_0506_F000_0000_0000_0000_0000_0000_0000_0000_0000
    dut.packed_a0.value = 0x1000_0000_0000_0000 #_0400_FDFC_0506_F000_0000_0000_0000_0000_0000_0000_0000_0000
    dut.packed_a1.value = 0x0000_0000_0000_0000 #_0400_FDFC_0506_F000_0000_0000_0000_0000_0000_0000_0000_0000
    dut.packed_a2.value = 0x0000_0000_0000_0000 #_0400_FDFC_0506_F000_0000_0000_0000_0000_0000_0000_0000_0000
    dut.packed_a3.value = 0x0000_0000_0000_0000 #_0400_FDFC_0506_F000_0000_0000_0000_0000_0000_0000_0000_0000

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    for i in range(30):
        print("i", i, "state", dut.state.value)
        print("accum      ", convert_dut_var(dut.accum))
        print("result     ", dut.result.value)
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    assert dut.out_v.value == 1
    #                              0     1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
    assert dut.packed_out.value == 0x0b1e_0630_f85e_f7de_0be8_08f6_06ea_04b2

    # # same again with RELU

    # dut.apply_relu.value = 1

    # dut.rst.value = 1
    # await RisingEdge(dut.clk)
    # dut.rst.value = 0
    # await RisingEdge(dut.clk)

    # for i in range(30):
    #     if dut.out_v.value:
    #         break
    #     await RisingEdge(dut.clk)

    # assert dut.out_v.value == 1
    # #                                0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
    # assert dut.packed_out.value == 0x16e5_0000_1642_0000_0ba3_1ca9_076e_11c3_0000_0000_0000_0000_0000_0000_0000_0000
