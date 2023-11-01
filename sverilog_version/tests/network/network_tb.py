import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

#add .. to path so we can import a common test 'util'
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tb_util import *

conv1d_state_to_str = StateIdToStr('conv1d.sv')
po2_conv1d_state_to_str = StateIdToStr('po2_conv1d.sv')
network_state_to_str = StateIdToStr('network.sv')

def dump_dut_values(id_strs, dut_objs, unpack=False,
                    emit_bin=False, emit_hex=True, emit_dec=True):
    assert len(id_strs) == len(dut_objs)

    if unpack:
        unpacked_values = [unpack_binary(v.value) for v in dut_objs]
    else:
        # already unpacked
        unpacked_values = [v.value for v in dut_objs]
    if emit_bin:
        for s, v in zip(id_strs, unpacked_values):
            print(s, "bin", v)

    hex_values = [list(map(bits_to_hex, v)) for v in unpacked_values]
    if emit_hex:
        for s, v in zip(id_strs, hex_values):
            print(s, "hex", v)

    # note: emitting also includes calculation here too.
    if emit_dec:
        dec_values = [list(map(hex_fp_value_to_decimal, v)) for v in hex_values]
        for s, v in zip(id_strs, dec_values):
            print(s, "dec", v)


@cocotb.test()
async def test_networks(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    print("FILTER_D", dut.FILTER_D.value)

    # slurp entries from test_x values
    test_x_hex_values = []
    with open('test_x.hex', 'r') as f:
        for line in f.readlines():
            inputs = line.strip().split(" ")
            assert len(inputs) == 4
            test_x_hex_values.append([eval(e) for e in inputs])
    print("READ", len(test_x_hex_values), "INPUTS")

    def clock_next_sample():
        next_x_vals = test_x_hex_values.pop(0)
        print("!clock_next_sample! next_x_vals", next_x_vals)
        dut.sample_in0.value = next_x_vals[0]
        dut.sample_in1.value = next_x_vals[1]
        dut.sample_in2.value = next_x_vals[2]
        dut.sample_in3.value = next_x_vals[3]
        dut.sample_clk.value = 1

    clock_next_sample()

    for i in range(100000):

        print("|test_x_hex_values|=", len(test_x_hex_values))

        try:
            if int(dut.state.value) == 12:  # OUTPUT
                if len(test_x_hex_values) == 0:
                    # we are done
                    break
                clock_next_sample()
        except ValueError as e:
            # just the fact dut.state.value not set yet (?)
            assert 'Unresolvable bit in binary string' in str(e)
            pass

        print("===i", i, "state", network_state_to_str[dut.state.value])

        print("c0 rst", dut.c0_rst.value,
               "out_v", dut.c0_out_v.value,
               "state", conv1d_state_to_str[dut.conv0.state])
        print("c0_out", convert_dut_var(unpack_binary(dut.c0_out)))

        print("c1 rst", dut.c1_rst.value,
              "out_v", dut.c1_out_v.value,
              "state", conv1d_state_to_str[dut.conv1.state])
        print("c1_out", convert_dut_var(unpack_binary(dut.c1_out)))

        try:
            print("c1_1a_rst", dut.c1_1a_rst.value,
                  "c1_1a_out_v", dut.c1_1a_out_v.value,
                  "state", po2_conv1d_state_to_str[dut.conv1_1a.state])
            print("c1_1a_out", convert_dut_var(unpack_binary(dut.c1_1a_out)))
            print("c1_1b_rst", dut.c1_1b_rst.value,
                  "c1_1b_out_v", dut.c1_1b_out_v.value,
                  "state", po2_conv1d_state_to_str[dut.conv1_1b.state])
            print("c1_2a_rst", dut.c1_2a_rst.value,
                  "c1_2a_out_v", dut.c1_2a_out_v.value,
                  "state", po2_conv1d_state_to_str[dut.conv1_2a.state])
            print("c1_2b_rst", dut.c1_2b_rst.value,
                  "c1_2b_out_v", dut.c1_2b_out_v.value,
                  "state", po2_conv1d_state_to_str[dut.conv1_2b.state])
        except AttributeError as e:
            # these don't exist for the qb version of the network.
            # should do this in cleaner way :/ e.g. check an env var?
            pass

        print("c2 rst", dut.c2_rst.value,
              "out_v", dut.c2_out_v.value,
              "state", conv1d_state_to_str[dut.conv2.state])

        print("----------- final output")

        out_bin = [dut.sample_out0.value, dut.sample_out1.value, dut.sample_out2.value, dut.sample_out3.value]
        #print("OUT  bin", out_bin)
        out_hex = list(map(bits_to_hex, out_bin))
        #print("OUT  hex", out_hex)
        out_dec = list(map(hex_fp_value_to_decimal, out_hex))
        print("OUT dec", " ".join(map(str, out_dec)))

        await RisingEdge(dut.clk)

        dut.sample_clk.value = 0