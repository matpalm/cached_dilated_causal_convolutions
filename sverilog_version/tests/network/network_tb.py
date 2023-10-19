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

# pull these here from fxputil since can't import fxputil when
# running under cocotb test ( different python env :/ )

n_int = 4

def _bit_not(n):
    return (1 << n_int) - 1 - n

def _twos_comp_to_signed(n):
    if (1 << (n_int-1) & n) > 0:
        return -int(_bit_not(n) + 1)
    else:
        return int(n)

def no_value_yet(s):
    s = str(s)
    return 'x' in s or 'z' in s

def hex_fp_value_to_decimal(hex_fp_str):
    if no_value_yet(hex_fp_str): return hex_fp_str
    assert type(hex_fp_str) == str
    value = int(hex_fp_str, 16)
    if len(hex_fp_str) == 4:
        integer_bits = value >> 12
        integer_value = _twos_comp_to_signed(integer_bits)
        fractional_bits = value & 0xFFF
        fractional_value = fractional_bits / float(2**12)
        return integer_value + fractional_value
    elif len(hex_fp_str) == 8:
        raise Exception("Fix this! not working for signed")
        integer_bits = value >> 24
        integer_value = _twos_comp_to_signed(integer_bits)
        fractional_bits = value & 0xFFFFFF
        fractional_value = fractional_bits / float(2**24)
        return integer_value + fractional_value
    else:
        raise Exception(len(hex_fp_str))

# def fixed_point_array_to_decimals(a):
#     if no_value_yet(a): return a
#     return [fixed_point_to_decimal(v) for v in a]

def bits_to_hex(value):
    value = str(value)
    if no_value_yet(value): return value
    value_len = len(value)
    value = int(value, 2)
    if value_len == 16:
        return f"{value:04x}"
    elif value_len == 32:
        return f"{value:08x}"
    else:
        raise Exception(value_len)

def unpack_binary(values, W=16):
    values = str(values)
    if no_value_yet(values): return values
    assert len(values) >= W and len(values) % W == 0
    results = []
    while len(values) > 0:
        results.append(values[:W])
        values = values[W:]
    return results

def conv_state_to_str(s):
    return {
        0: 'MAT_MUL_RUNNING',
        1: 'ACCUMULATE',
        2: 'BIAS_ADD',
        3: 'CLIP_LOWER',
        4: 'CLIP_UPPER',
        5: 'SINGLE_W',
        6: 'OUTPUT'
    }[s]

@cocotb.test()
async def test_networks(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    # slurp entries from test_x values
    test_x_hex_values = []
    with open('test_x.hex', 'r') as f:
        for line in f.readlines():
            inputs = line.strip().split(" ")
            assert len(inputs) == 4
            test_x_hex_values.append([eval(e) for e in inputs])
    print("READ", len(test_x_hex_values), "INPUTS")

    # fxpmath version is only running 200, so only do 200 here too
    test_x_hex_values = test_x_hex_values[:200]

    def clock_next_sample():
        next_x_vals = test_x_hex_values.pop(0)
        print("!clock_next_sample! next_x_vals", next_x_vals)
        dut.sample_in0.value = next_x_vals[0]
        dut.sample_in1.value = next_x_vals[1]
        dut.sample_in2.value = next_x_vals[2]
        dut.sample_in3.value = next_x_vals[3]
        dut.sample_clk = 1

    clock_next_sample()

    def net_state_to_str():
        if dut.state.value == 0:
            return "CLK_LSB"
        elif dut.state.value == 1:
            return "RST_CONV_0"
        elif dut.state.value == 2:
            return "CONV_0_RUNNING"
        elif dut.state.value == 3:
            return "CLK_ACT_CACHE_0"
        elif dut.state.value == 4:
            return "RST_CONV_1"
        elif dut.state.value == 5:
            return "CONV_1_RUNNING"
        elif dut.state.value == 6:
            return "CLK_ACT_CACHE_1"
        elif dut.state.value == 7:
            return "RST_CONV_2"
        elif dut.state.value == 8:
            return "CONV_2_RUNNING"
        elif dut.state.value == 9:
            return "CLK_ACT_CACHE_2"
        elif dut.state.value == 10:
            return "RST_CONV_3"
        elif dut.state.value == 11:
            return "CONV_3_RUNNING"
        elif dut.state.value == 12:
            return "OUTPUT"
        else:
            raise Exception(f"unknown state [{dut.state.value}]")

    for i in range(100000):

        print("|test_x_hex_values|=", len(test_x_hex_values))

        if int(dut.state.value) == 12:  # OUTPUT
            if len(test_x_hex_values) == 0:
                # we are done
                break
            clock_next_sample()

        print("===i", i, "state", net_state_to_str(), dut.state.value)

        print("dut.sample_in0.value", dut.sample_in0.value, bits_to_hex(dut.sample_in0.value))
        print("dut.sample_in1.value", dut.sample_in1.value, bits_to_hex(dut.sample_in1.value))
        print("dut.sample_in2.value", dut.sample_in2.value, bits_to_hex(dut.sample_in2.value))
        print("dut.sample_in3.value", dut.sample_in3.value, bits_to_hex(dut.sample_in3.value))

        print("----------- LSB")

        lsb_in0 = [dut.lsb_out_in0_0.value, dut.lsb_out_in0_1.value, dut.lsb_out_in0_2.value, dut.lsb_out_in0_3.value]
        lsb_in1 = [dut.lsb_out_in1_0.value, dut.lsb_out_in1_1.value, dut.lsb_out_in1_2.value, dut.lsb_out_in1_3.value]
        lsb_in2 = [dut.lsb_out_in2_0.value, dut.lsb_out_in2_1.value, dut.lsb_out_in2_2.value, dut.lsb_out_in2_3.value]
        lsb_in3 = [dut.lsb_out_in3_0.value, dut.lsb_out_in3_1.value, dut.lsb_out_in3_2.value, dut.lsb_out_in3_3.value]
        # print("lsb_in0 bin  ",  lsb_in0)
        # print("lsb_in1 bin  ",  lsb_in1)
        # print("lsb_in2 bin  ",  lsb_in2)
        # print("lsb_in3 bin  ",  lsb_in3)

        lsb_in0_hex = list(map(bits_to_hex, lsb_in0))
        lsb_in1_hex = list(map(bits_to_hex, lsb_in1))
        lsb_in2_hex = list(map(bits_to_hex, lsb_in2))
        lsb_in3_hex = list(map(bits_to_hex, lsb_in3))
        # print("lsb_in0 hex  ",  lsb_in0_hex)
        # print("lsb_in1 hex  ",  lsb_in1_hex)
        # print("lsb_in2 hex  ",  lsb_in2_hex)
        # print("lsb_in3 hex  ",  lsb_in3_hex)

        lsb_in0_dec = list(map(hex_fp_value_to_decimal, lsb_in0_hex))
        lsb_in1_dec = list(map(hex_fp_value_to_decimal, lsb_in1_hex))
        lsb_in2_dec = list(map(hex_fp_value_to_decimal, lsb_in2_hex))
        lsb_in3_dec = list(map(hex_fp_value_to_decimal, lsb_in3_hex))
        print("lsb_in0 dec  ",  lsb_in0_dec)
        print("lsb_in1 dec  ",  lsb_in1_dec)
        print("lsb_in2 dec  ",  lsb_in2_dec)
        print("lsb_in3 dec  ",  lsb_in3_dec)

        print("----------- conv0")

        print("c0_rst", dut.c0_rst.value)
        print("c0d_state", dut.conv0.state.value, conv_state_to_str(int(dut.conv0.state.value)))
        print("c0_out_v", dut.c0_out_v.value)

        # c0a0_u = unpack_binary(dut.c0a0.value)
        # c0a1_u = unpack_binary(dut.c0a1.value)
        # c0a2_u = unpack_binary(dut.c0a2.value)
        # c0a3_u = unpack_binary(dut.c0a3.value)
        # print("c0a0 bin  ", c0a0_u)
        # print("c0a1 bin  ", c0a1_u)
        # print("c0a2 bin  ", c0a2_u)
        # print("c0a3 bin  ", c0a3_u)

        # c0a0_h = list(map(bits_to_hex, c0a0_u))
        # c0a1_h = list(map(bits_to_hex, c0a1_u))
        # c0a2_h = list(map(bits_to_hex, c0a2_u))
        # c0a3_h = list(map(bits_to_hex, c0a3_u))
        # print("c0a0 hex  ", c0a0_h)
        # print("c0a1 hex  ", c0a1_h)
        # print("c0a2 hex  ", c0a2_h)
        # print("c0a3 hex  ", c0a3_h)

        # c0a0_d = list(map(hex_fp_value_to_decimal, c0a0_h))
        # c0a1_d = list(map(hex_fp_value_to_decimal, c0a1_h))
        # c0a2_d = list(map(hex_fp_value_to_decimal, c0a2_h))
        # c0a3_d = list(map(hex_fp_value_to_decimal, c0a3_h))
        # print("c0a0 dec  ", c0a0_d)
        # print("c0a1 dec  ", c0a1_d)
        # print("c0a2 dec  ", c0a2_d)
        # print("c0a3 dec  ", c0a3_d)

        # c0k0_b = dut.conv0.kernel0_out.value
        # c0k1_b = dut.conv0.kernel1_out.value
        # c0k2_b = dut.conv0.kernel2_out.value
        # c0k3_b = dut.conv0.kernel3_out.value
        # print("c0 k0 bin  ", c0k0_b)
        # print("c0 k1 bin  ", c0k1_b)
        # print("c0 k2 bin  ", c0k2_b)
        # print("c0 k3 bin  ", c0k3_b)

        # c0k0_h = list(map(bits_to_hex, c0k0_b))
        # c0k1_h = list(map(bits_to_hex, c0k1_b))
        # c0k2_h = list(map(bits_to_hex, c0k2_b))
        # c0k3_h = list(map(bits_to_hex, c0k3_b))
        # print("c0 k0 hex  ", c0k0_h)
        # print("c0 k1 hex  ", c0k1_h)
        # print("c0 k2 hex  ", c0k2_h)
        # print("c0 k3 hex  ", c0k3_h)

        # c0biases_h = list(map(bits_to_hex, dut.conv0.bias_values.value))
        # print("c0bias hex ", c0biases_h)

        # c0_accum_b = dut.conv0.accum.value
        # # print("c0_accum b  ", c0_accum_b)
        # c0_accum_h = list(map(bits_to_hex, c0_accum_b))
        # print("c0_accum h  ", c0_accum_h)

        c0_result_b = dut.conv0.result.value
        # print("c0_result b ", c0_result_b)
        c0_result_h = list(map(bits_to_hex, c0_result_b))
        # print("c0_result h ", c0_result_h)
        c0_result_d = list(map(hex_fp_value_to_decimal, c0_result_h))
        print("c0_result d ", c0_result_d)

        # TODO: conversion to FP for double width broken (?)
        # c0k0_dec = list(map(hex_fp_value_to_decimal, c0k0_hex))
        # c0k1_dec = list(map(hex_fp_value_to_decimal, c0k1_hex))
        # c0k2_dec = list(map(hex_fp_value_to_decimal, c0k2_hex))
        # c0k3_dec = list(map(hex_fp_value_to_decimal, c0k3_hex))
        # print("c0 k0 hex  ", c0k0_dec)
        # print("c0 k1 hex  ", c0k1_dec)
        # print("c0 k2 hex  ", c0k2_dec)
        # print("c0 k3 hex  ", c0k3_dec)

        c0_out_bin_values = unpack_binary(dut.c0_out.value)
        # print("c0_out  bin", c0_out_bin_values)
        c0_out_hex = list(map(bits_to_hex, c0_out_bin_values))
        # print("c0_out  hex", c0_out_hex)
        c0_out_dec = list(map(hex_fp_value_to_decimal, c0_out_hex))
        print("c0_out  dec", c0_out_dec)

        print("----------- ac0")

        print("ac_c0_clk", dut.ac_c0_clk.value)

        # print("----------- conv1")

        # print("c1_rst   ", dut.c1_rst.value)
        # print("c1_out_v ", dut.c1_out_v.value)

        # c1a0_u = unpack_binary(dut.c1a0.value)
        # c1a1_u = unpack_binary(dut.c1a1.value)
        # c1a2_u = unpack_binary(dut.c1a2.value)
        # c1a3_u = unpack_binary(dut.c1a3.value)
        # print("c1a0 bin  ", c1a0_u)
        # print("c1a1 bin  ", c1a1_u)
        # print("c1a2 bin  ", c1a2_u)
        # print("c1a3 bin  ", c1a3_u)

        # c1a0_dec = list(map(hex_fp_value_to_decimal, map(bits_to_hex, c1a0_u)))
        # c1a1_dec = list(map(hex_fp_value_to_decimal, map(bits_to_hex, c1a1_u)))
        # c1a2_dec = list(map(hex_fp_value_to_decimal, map(bits_to_hex, c1a2_u)))
        # c1a3_dec = list(map(hex_fp_value_to_decimal, map(bits_to_hex, c1a3_u)))
        # print("c1a0 dec  ", c1a0_dec)
        # print("c1a1 dec  ", c1a1_dec)
        # print("c1a2 dec  ", c1a2_dec)
        # print("c1a3 dec  ", c1a3_dec)

        # c1_out_bin_values = unpack_binary(dut.c1_out.value)
        # print("c1_out  bin", c1_out_bin_values)
        # c1_out_hex = list(map(bits_to_hex, c1_out_bin_values))
        # print("c1_out  hex", c1_out_hex)
        # c1_out_dec = list(map(hex_fp_value_to_decimal, c1_out_hex))
        # print("c1_out  dec", c1_out_dec)

        out_bin = [dut.sample_out0.value, dut.sample_out1.value, dut.sample_out2.value, dut.sample_out3.value]
        print("OUT  bin", out_bin)
        out_hex = list(map(bits_to_hex, out_bin))
        print("OUT  hex", out_hex)
        out_dec = list(map(hex_fp_value_to_decimal, out_hex))
        print("OUT dec", " ".join(map(str, out_dec)))

        await RisingEdge(dut.clk)

        dut.sample_clk = 0