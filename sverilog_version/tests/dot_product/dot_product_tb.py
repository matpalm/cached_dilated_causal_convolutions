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

    # dut.a_d0.value = 0x0400   # 0000.010000000000 0.25
    # dut.a_d1.value = 0xFDFC   # 1111.110111111100 -0.1259765625
    # dut.a_d2.value = 0x0506   # 0000.010100000110 0.31396484375
    # dut.a_d3.value = 0xF000   # 1111.000000000000 -1.0
    # dut.a_d4.value = 0x1000   # 0001.010000000000 1.0
    # dut.a_d5.value = 0
    # dut.a_d6.value = 0xF000   # 1111.000000000000 -1.0
    # dut.a_d7.value = 0x1000   # 0001.010000000000 1.0

    #                      0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
    dut.packed_a.value = 0x0400_FDFC_0506_F000_1000_0000_F000_1000_0000_0000_0000_0000_0000_0000_0000_1000

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # note: b values read from b_values.hex

    print("unpacked a?", dut.a.value)

    for i in range(20):
        if dut.out_v.value:
            break
        print("i", i, "waiting", dut.dp_state.value)
        print("dp.i    ", dut.i.value)
        try:
            print("dp.a[i] ", dut.a.value[dut.i.value])
            print("dp.b[i] ", dut.b_values.value[dut.i.value])
        except IndexError:
            pass
        print("acc0    ", dut.acc0.value)
        #print("acc1    ", dut.acc1.value)
        print("product0", dut.product0.value)
        #print("product1", dut.product1.value)
        await RisingEdge(dut.clk)

    # should be valid
    assert dut.out_v.value == 1

    assert dut.out.value == 0xfe_ecc15a
