import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

#add .. to path so we can import a common test 'util'
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tb_util import *

po2_multiply_idx_to_str = dict(enumerate(
    'NEGATE_1 NEGATE_2 PAD_TO_DOUBLE_WIDTH SHIFT EMIT_ZERO DONE'.split(' ')
    ))
def po2_multiply_state_to_str(s):
    as_str = po2_multiply_idx_to_str[int(s)]
    return f"{as_str} ({s})"

po2_dot_product_idx_to_str = dict(enumerate(
    'MULTIPLYING_ELEMENTS ADD_16 ADD_8 ADD_4 ADD_2 DONE'.split(' ')
    ))
def po2_dot_product_state_to_str(s):
    as_str = po2_dot_product_idx_to_str[int(s)]
    return f"{as_str} ({s})"

async def test_input_output(dut, input, expected_out):

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
    for i in range(10):
        print("===", i)
        print("state", po2_dot_product_state_to_str(dut.state.value))

        mN_states = [
            dut.m0.state.value, dut.m1.state.value,
            dut.m2.state.value, dut.m3.state.value]
        print("dp_states", [po2_multiply_state_to_str(s) for s in mN_states])

        print("result")
        r = convert_dut_var(dut.result)
        for e in r: print(e)

        print("result_v", dut.result_v.value)

        print("accumulators")
        r = convert_dut_var(dut.accumulator)
        for e in r: print(e)

        print("out   ", convert_dut_var(dut.out))
        print("out_v ", dut.out_v.value)

        if dut.out_v == 1: break

        await RisingEdge(dut.clk)

    _b, _h, d = convert_dut_var(dut.out)
    assert d == expected_out


@cocotb.test()
async def test_po2_dot_product(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    # zero input => zero output
    await test_input_output(dut, input = 0,  expected_out = 0)

    await test_input_output(dut,
        #          0    1    2    3
        #          1    1    2    1/2  0    0    0    0
        input = 0x1000_1000_2000_0800_0000_0000_0000_1000,
        expected_out = (0 - 1/2 + 2 + 1/32 + 1/2)
    )

