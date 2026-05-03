from pathlib import Path
import sys
import unittest
import numpy as np

from amaranth.sim import Simulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdcc import parse_nnq, K
from cdcc.activation_cache import ActivationCache


class TestActivationCache(unittest.TestCase):

    def test_activation_cache(self):
        dut = ActivationCache(in_out_d=3, dilation_level=1)

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            for i in range(20):
                sample = np.array([0.01, 0.02, 0.03])
                sample += i * 0.1
                sample_nnq = parse_nnq(sample, assert_exact=False)

                ctx.set(dut.i.payload, sample_nnq)
                ctx.set(dut.i.valid, 1)
                await ctx.tick()

                self.assertEqual(ctx.get(dut.o.valid), 1)
                # for k in range(K):
                #     for d in range(3):
                #         print("o", k, d, ctx.get(dut.o.payload[k][d]).as_float())

                def assert_almost_equals(k, expecteds):
                    expecteds = parse_nnq(expecteds, assert_exact=False)
                    for d in range(3):
                        actual = ctx.get(dut.o.payload[k][d]).as_float()
                        self.assertEqual(actual, expecteds[d].as_float())

                if i == 0:
                    # at first step all entries, but last, will be zero
                    for k in [0, 1, 2]:  # but not k=3, as checked
                        for d in range(3):
                            actual = ctx.get(dut.o.payload[k][d]).as_float()
                            self.assertEqual(actual, 0)
                    # and last will be the current sample
                    for d in range(3):
                        actual = ctx.get(dut.o.payload[3][d]).as_float()
                        expected = sample_nnq[d].as_float()
                        self.assertAlmostEqual(actual, expected)

                elif i == 19:
                    # at last step all entries, but last, will be zero
                    # entry from 4^3=64 steps ago
                    assert_almost_equals(k=0, expecteds=[0.81, 0.82, 0.83])
                    # entry from 4^2=16 steps ago
                    assert_almost_equals(k=1, expecteds=[1.21, 1.22, 1.23])
                    # entry from 4^1=4 steps ago
                    assert_almost_equals(k=2, expecteds=[1.61, 1.62, 1.63])
                    # most recent entry
                    assert_almost_equals(k=3, expecteds=[1.91, 1.92, 1.93])

                else:
                    # in general, at every step the last entry is the latest added
                    for d in range(3):
                        actual = ctx.get(dut.o.payload[3][d]).as_float()
                        expected = sample_nnq[d].as_float()
                        self.assertAlmostEqual(actual, expected)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()
