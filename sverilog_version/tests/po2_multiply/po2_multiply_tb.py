import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

#add .. to path so we can import a common test 'util'
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tb_util import *

async def test(dut, input, negative_weight, log2_weight, expected_result):
    # inp=3.0 * weight=0.25 = 0.75

    dut.inp.value = input
    dut.negative_weight.value = negative_weight
    dut.log_2_weight.value = log2_weight

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    for i in range(10):
        print("state", dut.state.value)
        print("result_v", dut.result_v.value)
        print("result", convert_dut_var(dut.result))
        if dut.result_v.value == 1: break
        await RisingEdge(dut.clk)

    assert dut.result_v.value == 1

    fixed_point_result = dut.result.value
    return hex_fp_value_to_decimal(bits_to_hex(fixed_point_result))


@cocotb.test()
async def test_ad_hoc_test(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    # inp=3.0 * weight=0.25 = 0.75
    await test(dut, input=0x3000, negative_weight=0, log2_weight=2, expected_result=0.75)

    # inp=3.0 * weight=-0.25 = -0.75
    await test(dut, input=0x3000, negative_weight=1, log2_weight=2, expected_result=-0.75)

    # inp=-3.0 * weight=0.25 = -0.75
    await test(dut, input=0xd000, negative_weight=1, log2_weight=2, expected_result=-0.75)

    # inp=-3.0 * weight=-0.25 = 0.75
    await test(dut, input=0xd000, negative_weight=1, log2_weight=2, expected_result=0.75)
