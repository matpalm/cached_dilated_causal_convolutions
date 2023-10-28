import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

#add .. to path so we can import a common test 'util'
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tb_util import *

def po_multiply_state_to_str(s):
    as_str = {
        0: 'NEGATE_1',
        1: 'NEGATE_2',
        2: 'PAD_TO_DOUBLE_WIDTH',
        3: 'SHIFT',
        4: 'DONE',
    }[int(s)]
    return f"{as_str} ({s})"

def po_dot_product_state_to_str(s):
    as_str = {
        0: 'MULTIPLYING_ELEMENTS',
        1: 'ADD_1',
        2: 'ADD_2',
        3: 'DONE',
    }[int(s)]
    return f"{as_str} ({s})"

@cocotb.test()
async def test_po2_dot_product(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    #                      0    1    2    3
    dut.packed_a.value = 0x1000_1000_2000_0100

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    print("packed_a    ", dut.packed_a.value)

    print("zero_weights", dut.zero_weights.value)          # 1 0 0 0
    print("negative_weights", dut.negative_weights.value)  # 0 1 0 0
    print("log_2_weights", dut.log_2_weights.value)        # 1 1 0 4

    # DP is then
    # e0 = 0 ( since zero weight=0 )
    # e1 = -1 >> 1 = -1 / 2 = -1/2
    # e2 = 2 >> 0 = 2
    # e3 = 1/2 >> 4 = 1/32
    # sum = 1.46875
    for i in range(10):
        print("===", i)
        print("state", po_dot_product_state_to_str(dut.state.value))

        mN_states = [
            dut.m0.state.value, dut.m1.state.value,
            dut.m2.state.value, dut.m3.state.value]
        print("dp_states", [po_multiply_state_to_str(s) for s in mN_states])

        print("result", convert_dut_var(dut.result))
        print("result_v", dut.result_v.value)
        print("out   ", convert_dut_var(dut.out))
        print("out_v ", dut.out_v.value)
        await RisingEdge(dut.clk)

    raise Exception("NEED TO ADD ASSERTIONS!")