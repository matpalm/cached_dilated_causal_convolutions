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

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    # zero in always => zero out
    await test_input_output(dut,
        input=0,
        expected_out=0)

    # TODO: investigate why, after the po2 fix re: negation of entire number ( not just
    #       integer part ) , that these numbers were off by just a bit..
    #       e.g. before we had -0.5 (ff800000) but now it's -0.4999995827674866 (ff800007)

    # input of 1,0,0,0 selects first element from each column
    # in this data case the last element is an example of zero weight
    await test_input_output(dut,
        #       0    1    2    3
        input=0x1000_0000_0000_0000,
        #              0        1        2        3        4        5        6        7
        #              +1/8     ~ -1     +1/2     ~ -1/4   +1/4     +1       +1/2     0
        expected_out=0x00200000_ff00000f_00800000_ffc00003_00400000_01000000_00800000_00000000
    )

    # input of 0,1,0,0 selects second element from each column
    # in this example elements 5 and 6 have zero weight
    await test_input_output(dut,
        #       0    1    2    3
        input=0x0000_1000_0000_0000,
        #              0        1        2        3        4        5        6        7
        #              ~ 1/8    +1/2     +1/2     ~ -1/2   +1/2     0        0        -1/32
        expected_out=0xffe00001_00800000_00800000_ff800007_00800000_00000000_00000000_fff80000
    )

    # input of 1,1,0,0 selects first and second element from each column and sums them
    await test_input_output(dut,
        #       0    1    2    3
        input=0x1000_1000_0000_0000,
        #              0        1        2        3        4        5        6        7
        #              ~ 0      ~ -1/2   +1       ~ -3/4   +3/4     +1       +1/2     -1/32
        expected_out=0x00000001_ff80000f_01000000_ff40000a_00c00000_01000000_00800000_fff80000
    )
