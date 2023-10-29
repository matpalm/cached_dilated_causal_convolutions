import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, FallingEdge, RisingEdge, ClockCycles
from cocotb.handle import Force, Release

#add .. to path so we can import a common test 'util'
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tb_util import *

@cocotb.test()
async def test_ad_hoc_test(dut):

    clock = Clock(dut.clk, 83, units='ns')
    cocotb.start_soon(clock.start())

    dut.inp1.value = 0x7001;
    dut.inp2.value = 0

    for i in range(2):
        await RisingEdge(dut.clk)

    # print("clamp_l", dut.CLAMPL.value)
    # print("clamp_h", dut.CLAMPH.value)

    print("inp1", convert_dut_var(dut.inp1))
#    print("inp2", dut.inp2.value)
    print("out1", convert_dut_var(dut.out1))
#    print("out2", dut.out2.value)
