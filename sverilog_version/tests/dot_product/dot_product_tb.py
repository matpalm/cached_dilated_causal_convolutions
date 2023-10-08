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
async def test_1d_dot_product_low_values(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    dut.a.value = [
        0x0400,   # 0000.010000000000 0.25
        0xFDFC,   # 1111.110111111100 -0.1259765625
        0x0506,   # 0000.010100000110 0.31396484375
        0xF000,   # 1111.000000000000 -1.0
        # 0x0400,   # 0000.010000000000 0.25
        # 0xF000,   # 1111.000000000000 -1.0
        # 0x0028,   # 0000.000000101000 0.009765625
        # 0xFFAF,   # 1111.111110101111 -0.019775390625
    ]

    dut.b.value = [
        0xF3D8,  # 1111.001111011000 -0.759765625
        0x0999,  # 0000.100110011001 0.599853515625
        0xFD75,  # 1111.110101110101 -0.158935546875
        0x1000,  # 0001.000000000000 1.0
    ]
    # note: b values read from b_values.hex

    for i in range(10):
        if dut.out_v.value:
            break
        # print("i", i, "waiting", dut.dp_state.value)
        # print("acc0    ", dut.acc0.value)
        # print("acc1    ", dut.acc1.value)
        # print("product0", dut.product0.value)
        # print("product1", dut.product1.value)
        await RisingEdge(dut.clk)

    # should be valid
    assert dut.out_v.value == 1

    # for i in range(8):
    #     print("B_values", i, dut.b_values[i])

    # required some minor rounding
    # dump(-0.5751953125-(2**-10)+(2**-11)+(2**-12))
    assert dut.out.value == 0xFEAF415A   # 1111 1110 1010 1111 0100 0001 0101 1010
