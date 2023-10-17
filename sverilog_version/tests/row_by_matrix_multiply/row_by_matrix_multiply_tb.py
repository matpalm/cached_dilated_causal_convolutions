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
async def test_row_by_matrix_multiply(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    dut.packed_a.value = 0

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)  # starts calculation

    for i in range(20):
        print("i", i,
              "col0_v", dut.col0_v.value,
              "col1_v", dut.col1_v.value,
              "col2_v", dut.col2_v.value,
              "col3_v", dut.col3_v.value)
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    assert dut.out_v.value == 1
    assert dut.out_d0.value == 0
    assert dut.out_d1.value == 0
    assert dut.out_d2.value == 0
    assert dut.out_d3.value == 0
    assert dut.out_d4.value == 0
    assert dut.out_d5.value == 0
    assert dut.out_d6.value == 0
    assert dut.out_d7.value == 0

    # set new values for a
    # dut.a_d0.value = 0x0400   # 0000.010000000000 0.25
    # dut.a_d1.value = 0xFDFC   # 1111.110111111100 -0.1259765625
    # dut.a_d2.value = 0x0506   # 0000.010100000110 0.31396484375
    # dut.a_d3.value = 0xF000   # 1111.000000000000 -1.0
    # dut.a_d4.value = 0x0400   # 0000.010000000000 0.25
    # dut.a_d5.value = 0xFDFC   # 1111.110111111100 -0.1259765625
    # dut.a_d6.value = 0x0506   # 0000.010100000110 0.31396484375
    # dut.a_d7.value = 0xF000   # 1111.000000000000 -1.0
    dut.packed_a.value = 0x0400_FDFC_0506_F000_0400_FDFC_0506_F000

    # trigger new run
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)  # starts calculation

    for i in range(20):
        print("i", i,
              "col0_v", dut.col0_v.value,
              "col1_v", dut.col1_v.value,
              "col2_v", dut.col2_v.value,
              "col3_v", dut.col3_v.value)
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    # should be valid
    assert dut.out_v.value == 1

    assert dut.out_d0.value == 0b11111101010111101000001010110100
    assert dut.out_d1.value == 0b11111110111000000100000000000000
    assert dut.out_d2.value == 0b11111101010111101000001010110100
    assert dut.out_d3.value == 0            # col3 weights are all zeros
    assert dut.out_d4.value == 0            # col3 weights are all zeros
    assert dut.out_d5.value == 0            # col3 weights are all zeros
    assert dut.out_d6.value == 0            # col3 weights are all zeros
    assert dut.out_d7.value == 0            # col3 weights are all zeros
