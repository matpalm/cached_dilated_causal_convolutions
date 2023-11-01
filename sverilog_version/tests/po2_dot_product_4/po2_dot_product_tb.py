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

    for i in range(100):

        print("===", i)
        print("state", po2_dot_product_state_to_str[dut.state.value])

        mN_states = [
            dut.m0.state.value, dut.m1.state.value,
            dut.m2.state.value, dut.m3.state.value]
        print("dp_states", [po2_multiply_state_to_str[s] for s in mN_states])

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

    assert dut.out_v == 1

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

    # test 2; a mix of values
    await test_input_output(dut,
        input=0x1000_1000_2000_0800,
        expected_out=(0 - 1/2 + 2 + 1/32))
