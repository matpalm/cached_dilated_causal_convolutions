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

    dut.apply_relu = 1

    dut.a0.value = [
        0x0400,   # 0000.010000000000 0.25
        0xFDFC,   # 1111.110111111100 -0.1259765625
        0x0506,   # 0000.010100000110 0.31396484375
        0xF000    # 1111.000000000000 -1.0
    ]
    dut.a1.value = [
        0x0400,   # 0000.010000000000 0.25
        0xFDFC,   # 1111.110111111100 -0.1259765625
        0x0506,   # 0000.010100000110 0.31396484375
        0xF000    # 1111.000000000000 -1.0
    ]
    dut.a2.value = [
        0x0400,   # 0000.010000000000 0.25
        0xFDFC,   # 1111.110111111100 -0.1259765625
        0x0506,   # 0000.010100000110 0.31396484375
        0xF000    # 1111.000000000000 -1.0
    ]
    dut.a3.value = [
        0x0400,   # 0000.010000000000 0.25
        0xFDFC,   # 1111.110111111100 -0.1259765625
        0x0506,   # 0000.010100000110 0.31396484375
        0xF000    # 1111.000000000000 -1.0
    ]

    # dut.a0.value = [
    #     0,
    #     0x0800,   # 0000.100000000000 0.5
    #     0x0400,   # 0000.010000000000 0.25
    #     0x0200   # 0000.001000000000 0.125
    # ]
    # dut.a1.value = [
    #     0,
    #     0xF800,   # 1111.100000000000 -0.5
    #     0xFC00,   # 1111.110000000000 -0.25
    #     0xFE00   # 1111.111000000000 -0.125
    # ]
    # dut.a2.value = [
    #     0,
    #     0x0800,   # 0000.100000000000 0.5
    #     0x0400,   # 0000.010000000000 0.25
    #     0x0200   # 0000.001000000000 0.125
    # ]
    # dut.a3.value = [
    #     0,
    #     0xF800,   # 1111.100000000000 -0.5
    #     0xFC00,   # 1111.110000000000 -0.25
    #     0xFE00   # 1111.111000000000 -0.125
    # ]

    # dut.a0.value = [
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000    # 1111.000000000000 -1.0
    # ]
    # dut.a1.value = [
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000    # 1111.000000000000 -1.0
    # ]
    # dut.a2.value = [
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000    # 1111.000000000000 -1.0
    # ]
    # dut.a3.value = [
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000,   # 1111.000000000000 -1.0
    #     0xF000    # 1111.000000000000 -1.0
    # ]

    # dut.a0.value = [
    #     0x1000,   # 1.0
    #     0x1000,   # 1.0
    #     0x1000,   # 1.0
    #     0x1000    # 1.0
    # ]
    # dut.a1.value = [
    #     0x1000,   # 1.0
    #     0x1000,   # 1.0
    #     0x1000,   # 1.0
    #     0x1000    # 1.0
    # ]
    # dut.a2.value = [
    #     0x1000,   # 1.0
    #     0x1000,   # 1.0
    #     0x1000,   # 1.0
    #     0x1000    # 1.0
    # ]
    # dut.a3.value = [
    #     0x1000,   # 1.0
    #     0x1000,   # 1.0
    #     0x1000,   # 1.0
    #     0x1000    # 1.0
    # ]

    # dut.a2.value = [
    #     0, 0, 0, 0
    # ]
    # dut.a3.value = [
    #     0, 0, 0, 0
    # ]

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)  # starts calculation

    for i in range(10):
        print("i", i, "c1d_state", dut.c1d_state.value)
        print("accum      ", dut.accum.value)
        print("result     ", dut.result.value)
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    assert dut.out_v.value == 1
    print("dut.out    ", dut.out.value)

    # # set new values for a
    # dut.a.value = [
    #     0x0400,   # 0000.010000000000 0.25
    #     0xFDFC,   # 1111.110111111100 -0.1259765625
    #     0x0506,   # 0000.010100000110 0.31396484375
    #     0xF000,   # 1111.000000000000 -1.0
    #     # 0x0400,   # 0000.010000000000 0.25
    #     # 0xF000,   # 1111.000000000000 -1.0
    #     # 0x0028,   # 0000.000000101000 0.009765625
    #     # 0xFFAF,   # 1111.111110101111 -0.019775390625
    # ]

    # # trigger new run
    # dut.rst.value = 1
    # await RisingEdge(dut.clk)
    # dut.rst.value = 0
    # await RisingEdge(dut.clk)  # starts calculation

    # for i in range(10):
    #     print("i", i,
    #           "col0_v", dut.col0_v.value,
    #           "col1_v", dut.col1_v.value,
    #           "col2_v", dut.col2_v.value,
    #           "col3_v", dut.col3_v.value)
    #     if dut.out_v.value:
    #         break
    #     await RisingEdge(dut.clk)

    # # should be valid
    # assert dut.out_v.value == 1

    # # print("dut.out[0].value", dut.out[0].value)
    # # print("dut.out[1].value", dut.out[1].value)
    # # print("dut.out[2].value", dut.out[2].value)
    # # print("dut.out[3].value", dut.out[3].value)

    # assert dut.out[0].value == 0xFEAF415A   # 1111 1110 1010 1111 0100 0001 0101 1010
    # assert dut.out[1].value == 0xFF702000   # 1111 1111 0111 0000 0010 0000 0000 0000
    # assert dut.out[2].value == 0xFEAF415A   # same as col0
    # assert dut.out[3].value == 0            # col3 weights are all zeros
