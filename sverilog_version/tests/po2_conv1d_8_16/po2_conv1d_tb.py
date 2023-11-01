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

        print("kernel0.col0.state", dut.kernel0.col0.state.value)
        print_value_per_line("kernel0.col0.result", dut.kernel0.col0.result)
        print_value_per_line("kernel0.col0.accumulator", dut.kernel0.col0.accumulator)

        print("kernel0.col_v", dut.kernel0.col_v.value)
        print("kernel_out", dut.kernel_out.value)
        # print_value_per_line("kernel_out_unpacked", dut.kernel_out_unpacked)
        # print_value_per_line("conv1d accum", dut.accum)
        # print_value_per_line("conv1d result", dut.result)
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
    # await test_input_output(dut,
    #     input=0,
    #     apply_relu=0,
    #     expected_out=0x01ee_fffa_0136_fec6_ffe8_fde4_011a_00d2_ffea_fef6_0064_038a_0054_ff06_fe2e_0040
    # )

    # real example from trying to match fxp math version
    await test_input_output(dut,
        input=0x0036_1398_0000_0000_0000_0000_053d_0000,
        apply_relu=0,
        expected_out=0x0029_0c2e_05e0_f39d_0919_0f13_0ba8_fadd_0b0c_0185_f63d_ff97_0ee5_09af_e7fe_0428
    )

# from FXP
# running layer relu
# result  [0.01318359375, 1.224609375, 0.0, 0.0, 0.0, 0.0, 0.327392578125, 0.0]
# running layer qconv_1_1a_po2 zero_weights=(1, 16, 8) negative_weights=(1, 16, 8) weights_log2=(1, 16, 8)
# row_by_matrix_multiply inputs
# cNaI i=0 (8,) [0.01318359375  1.224609375    0.             0.
#  0.             0.             0.327392578125 0.            ]
# KERNEL OUTPUTS
# kernel i=0 ['ffe3b600', '00c34800', '004aa000', 'ff4d7400',
#             '00931e00', '0112f800', '00a8ea00', 'ffa0b400',
#             '00b22000', '0028f800', 'ff5d9a00', 'ffc0d000',
#             '00e91000', '00aa9a00', 'fe9d0400', '003e8800']
# accumulated outputs ['0029', '0c2e', '05e0', 'f39e',
#                      '0919', '0f13', '0ba8', 'fade',
#                      '0b0c', '0185', 'f63e', 'ff97',
#                      '0ee5', '09af', 'e7ff', '0428']
# result  [0.010009765625, 0.76123046875, 0.3671875, -0.77392578125,
#          0.568603515625, 0.942138671875, 0.728515625, -0.32080078125,
#          0.6904296875, 0.094970703125, -0.60986328125, -0.025634765625,
#          0.930908203125, 0.605224609375, -1.500244140625, 0.259765625]

# PO2
# ( for input in hex )
# c1_out [('0000000000110110', '0036', 0.01318359375), ('0001001110011000', '1398', 1.224609375),
#     ('0000000000000000', '0000', 0.0), ('0000000000000000', '0000', 0.0),
#     ('0000000000000000', '0000', 0.0), ('0000000000000000', '0000', 0.0),
#     ('0000010100111101', '053d', 0.327392578125), ('0000000000000000', '0000', 0.0)]

