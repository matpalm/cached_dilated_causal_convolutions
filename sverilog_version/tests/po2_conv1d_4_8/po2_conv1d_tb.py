import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

# add .. to path so we can import a common test 'util'
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tb_util import convert_dut_var, print_value_per_line, unpack_binary, StateIdToStr

po2_conv1d_state_to_str = StateIdToStr('po2_conv1d.sv')

async def test_input_output(dut, input, apply_relu, expected_out):

    dut.apply_relu.value = apply_relu
    dut.packed_a.value = input

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)  # starts calculation

    print_value_per_line("bias_values", dut.bias_values)

    for i in range(20):
        print("-"*100)
        print("===i", i, po2_conv1d_state_to_str[dut.state.value])
        print("kernel0.col_v", dut.kernel0.col_v.value)
        print("kernel_out", dut.kernel_out.value)
        print_value_per_line("kernel_out_unpacked", dut.kernel_out_unpacked)
        print_value_per_line("accum", dut.accum)
        print_value_per_line("result", dut.result)
        print("packed_out", dut.packed_out.value)
        print("out_v", dut.out_v.value)
        if dut.out_v.value:
            break
        await RisingEdge(dut.clk)

    print_value_per_line("unpacked out", unpack_binary(dut.packed_out.value))

    assert dut.out_v.value == 1
    assert dut.packed_out.value == expected_out


@cocotb.test()
async def test_conv1d(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    # with zero input, and no relu, expected output is the bias values (single width)
    await test_input_output(dut,
        input=0,
        apply_relu=0,
        expected_out=0x1000_f800_0400_0000_1000_f800_0400_0000
    )

    # with zero input, and relu, expected output is the bias values (single width)
    # but with zero for any negative ones
    await test_input_output(dut,
        input=0,
        apply_relu=1,
        expected_out=0x1000_0000_0400_0000_1000_0000_0400_0000
    )

    # with (0,0,0,1) input the conv1d is just a mat mul with bias
    await test_input_output(dut,
        input=0x0000_0000_0000_1000,
        apply_relu=0,
        expected_out=0x1400_f600_fc00_f800_2000_f600_0000_1000
    )

    # with (0,0,0,1) input the conv1d is just a mat mul with bias
    await test_input_output(dut,
        input=0x0000_0000_0000_1000,
        apply_relu=1,
        expected_out=0x1400_0000_0000_0000_2000_0000_0000_1000
    )

