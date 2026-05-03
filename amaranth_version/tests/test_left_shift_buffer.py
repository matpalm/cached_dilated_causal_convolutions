from pathlib import Path
import sys
import unittest

from amaranth.sim import Simulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amaranth_future import fixed

from cdcc.left_shift_buffer import LeftShiftBuffer
from cdcc import NNQ, parse_nnq

class TestLeftShiftBuffer(unittest.TestCase):

    def test_left_shift_buffer(self):

        dut = LeftShiftBuffer()

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            inputs = [
                [0.11, 0.12, 0.13, 0],
                [0.21, 0.22, 0.23, 0],
                [0.31, 0.32, 0.33, 0],
                [0.41, 0.42, 0.43, 0],
                [0.51, 0.52, 0.53, 0],
                [0.61, 0.62, 0.63, 0],
            ]
            inputs = parse_nnq(inputs, assert_exact=False)

            for i, inp in enumerate(inputs):

                # set next input
                ctx.set(dut.i.payload, inputs[i])
                ctx.set(dut.i.valid, 1)
                await ctx.tick()

                # for first step we expect the first 3 elements to all
                # be zero and the last element to be the latest input
                if i == 0:
                    for k in range(4):
                        for out_d in range(4):
                            actual = ctx.get(dut.o.payload[k][out_d]).as_float()
                            if k != 3:
                                expected = 0
                            else:
                                expected = inputs[i][out_d].as_float()
                            self.assertEqual(actual, expected)

                # and by the time we get to last step we should see the last entry
                # in the kernel be the most recent, and each other one lagging
                if i == 5:
                    for k in range(4):
                        for out_d in range(4):
                            actual = ctx.get(dut.o.payload[k][out_d]).as_float()
                            expected = inputs[k + 2][out_d].as_float()
                            self.assertEqual(actual, expected)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        with sim.write_vcd(vcd_file=open("test_lsb.vcd", "w")):
            sim.run()
