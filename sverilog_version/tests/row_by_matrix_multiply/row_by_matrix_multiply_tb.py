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
              "col0_v", dut.col_v[0].value,
              "packed_out", dut.packed_out.value)
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    assert dut.out_v.value == 1
    assert dut.packed_out.value == 0

    # set new values for a
    # dut.a_d0.value = 0x0400   # 0000.010000000000 0.25
    # dut.a_d1.value = 0xFDFC   # 1111.110111111100 -0.1259765625
    # dut.a_d2.value = 0x0506   # 0000.010100000110 0.31396484375
    # dut.a_d3.value = 0xF000   # 1111.000000000000 -1.0
    # dut.a_d4.value = 0x0400   # 0000.010000000000 0.25
    # dut.a_d5.value = 0xFDFC   # 1111.110111111100 -0.1259765625
    # dut.a_d6.value = 0x0506   # 0000.010100000110 0.31396484375
    # dut.a_d7.value = 0xF000   # 1111.000000000000 -1.0

    #                      0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
    dut.packed_a.value = 0x0000_0100_0000_0000 #_0400_FDFC_0506_F000_0000_0000_0000_0000_0000_0000_0000_1000

    # trigger new run
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)  # starts calculation

    for i in range(20):
        print("i", i,
              "col0_v", dut.col_v[0].value,
              "packed_out", dut.packed_out.value)
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    # should be valid
    assert dut.out_v.value == 1

    # should pick second elements from each col
    #                               d0         d1       d2       d3       d4       d5       d6       d7       d8       d9       d10      d11      d12      d13      d14      d15
    #assert dut.packed_out.value == 0xfe5e82b4_fee04000_fd5e82b4_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000_00100000
    #assert dut.packed_out.value == 0xff3d8000_01000000_ff3d8000_00000000_00000000_00000000_00000000_00000000
    assert dut.packed_out.value ==  0x00099900_00100000_00099900_00000000_00000000_00000000_00000000_00000000