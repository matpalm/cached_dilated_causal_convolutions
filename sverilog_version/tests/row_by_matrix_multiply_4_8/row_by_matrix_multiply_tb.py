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

async def test_input_output(dut, input, expected_out):

    dut.packed_a.value = input

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
    assert dut.packed_out.value == expected_out

@cocotb.test()
async def test_row_by_matrix_multiply(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    # zero in => zero out
    await test_input_output(dut, input=0, expected_out=0)

    # more general test
    await test_input_output(dut,
        input=0x0000_0100_0000_0000,
        expected_out=0x00099900_00100000_00099900_00000000_00000000_00000000_00000000_00000000)
