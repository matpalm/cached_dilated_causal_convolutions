import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

#add .. to path so we can import a common test 'util'
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tb_util import *

po2_dot_product_state_to_str = StateIdToStr('po2_dot_product.sv')

async def test_input_output(dut, input, expected_out):
    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    # 0 input => 0 output

    dut.packed_a.value = input

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)  # starts calculation

    for i in range(100):

        print("===============", i)

        print("packed_a", dut.packed_a)

        print("col0 state", po2_dot_product_state_to_str[dut.col0.state.value])
        print("col0 result", convert_dut_var(dut.col0.result))
        print("col0 accumulator", convert_dut_var(dut.col0.accumulator))

        for i, val in enumerate(convert_dut_var(dut.dp_N_out)):
            print("dp_N_out", i, val)

        print("col_v", dut.col_v.value)
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    assert dut.out_v.value == 1
    assert dut.packed_out.value == expected_out


@cocotb.test()
async def test_row_by_matrix_multiply(dut):

    # zero in always => zero out
    await test_input_output(dut,
        input=0,
        expected_out=0)

    # input of 0,0,1,0,0,0,0,0 selects first weight from each column
    # note: col 0 has zero_weights for d2
    await test_input_output(dut,
        #       0    1    2    3    4    5    6    7
        input=0x0000_0000_1000_0000_0000_0000_0000_0000,
        #              0        1        2        3
        #              0        ~ -1/4   +1/2     ~ -1/4
        expected_out=0x00000000_ffc00003_00800000_ffc00003
    )
