import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

#add .. to path so we can import a common test 'util'
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tb_util import *

po2_multiply_state_to_str = StateIdToStr('po2_multiply.sv')
po2_dot_product_state_to_str = StateIdToStr('po2_dot_product.sv')

async def test_input_output(dut, input, expected_out, atol=1e-5):

    dut.packed_a.value = input

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    print("packed_a    ", dut.packed_a.value)
    print("zero_weights", dut.zero_weights.value)          # 1 0 0 0
    print("negative_weights", dut.negative_weights.value)  # 0 1 0 0
    print("log_2_weights", dut.log_2_weights.value)        # 1 1 0 4

    # DP is then
    # e0 = 1 * 0 = 0 ( zero weight )
    # e1 = 1 * -1 = -1 >> 1 = -1 / 2 = -1/2
    # e2 = 2 >> 0 = 2
    # e3 = 1/2 >> 4 = 1/32
    # sum = 1.46875
    for i in range(100):
        print("===", i)
        print("state         ", po2_dot_product_state_to_str[dut.state.value])
        print("po2 mult state", po2_multiply_state_to_str[dut.po2_mult.state.value])
        print("i", dut.i.value)
        print("accumulators", convert_dut_var(dut.accumulator))
        print("out   ", convert_dut_var(dut.out))
        print("out_v ", dut.out_v.value)

        if dut.out_v == 1: break

        await RisingEdge(dut.clk)

    _b, _h, d = convert_dut_var(dut.out)
    difference = d - expected_out
    if difference < atol:
        pass
    else:
        assert d == expected_out, "failed by tolerance diff"

@cocotb.test()
async def test_po2_dot_product(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    # zero input => zero output
    await test_input_output(dut, 0, 0)

    await test_input_output(dut,
        #       0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
        #       1    1    2    1/2  0    0    0    0
        input=0x1000_1000_2000_0800_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_1000,
        expected_out=(0 - 1/2 + 2 + 1/32 - 1/2)
        )
