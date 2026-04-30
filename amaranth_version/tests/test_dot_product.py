from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaranth.sim import Simulator

from amaranth_future import fixed

from cdcc import NNQ
from cdcc.dot_product import DotProduct


class TestDotProduct(unittest.TestCase):

    def test_dot_product_single_vector(self):

        weights = [0.5, -0.24, 0.125, -0.4]
        weights = [fixed.Const(w, shape=NNQ) for w in weights]

        dut = DotProduct(weights)

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            inp = [0.85, -0.5, 0.25, 0.125]
            inp = [fixed.Const(v, shape=NNQ) for v in inp]
            print("inp", [i.as_float() for i in inp])

            ctx.set(dut.i.payload, inp)
            ctx.set(dut.i.valid, 1)
            await ctx.tick()
            ctx.set(dut.i.valid, 0)

            for _ in range(dut.D + 3):
                if ctx.get(dut.o.valid):
                    break
                await ctx.tick()

            self.assertEqual(ctx.get(dut.o.valid), 1)

            actual = ctx.get(dut.o.payload).as_float()
            print("actual", actual)
            expected = sum(a.as_float() * b.as_float() for a, b in zip(inp, weights))
            print("expected", expected)
            self.assertAlmostEqual(actual, expected)

            await ctx.tick()
            self.assertEqual(ctx.get(dut.o.valid), 0)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()
