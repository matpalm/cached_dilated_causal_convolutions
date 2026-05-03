from pathlib import Path
import sys
import unittest
import pickle

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaranth.sim import Simulator

from cdcc import parse_nnq
from cdcc.qb_network_simple import QbNetworkSimple


class TestQbNetworkSimple(unittest.TestCase):

    def test_lsb_then_conv1d_single_output(self):
        weights = np.zeros((4, 1, 4), dtype=np.float64)
        # kernel index 3 corresponds to the current sample in the left-shift window
        weights[3, 0, 0] = 1.0

        dut = QbNetworkSimple(weights, apply_relu=False)

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            inp = parse_nnq([1.5, 0.0, 0.0, 0.0])
            zero_inp = parse_nnq([0.0, 0.0, 0.0, 0.0])

            # First transaction primes the left-shift buffer.
            ctx.set(dut.i.payload, inp)
            ctx.set(dut.i.valid, 1)
            await ctx.tick()
            ctx.set(dut.i.valid, 0)

            for _ in range(24):
                if ctx.get(dut.o.valid):
                    break
                await ctx.tick()

            self.assertEqual(ctx.get(dut.o.valid), 1)
            self.assertAlmostEqual(ctx.get(dut.o.payload).as_float(), 0.0)

            await ctx.tick()
            self.assertEqual(ctx.get(dut.o.valid), 0)

            # Second transaction makes Conv1d consume the primed window.
            ctx.set(dut.i.payload, zero_inp)
            ctx.set(dut.i.valid, 1)
            await ctx.tick()
            ctx.set(dut.i.valid, 0)

            for _ in range(24):
                if ctx.get(dut.o.valid):
                    break
                await ctx.tick()

            self.assertEqual(ctx.get(dut.o.valid), 1)
            self.assertAlmostEqual(ctx.get(dut.o.payload).as_float(), 1.5)

            await ctx.tick()
            self.assertEqual(ctx.get(dut.o.valid), 0)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()

    def test_qkeras_parse(self):
        with open("runs/41_tiliqua_1layer/weights/qkeras/latest.pkl", "rb") as f:
            d = pickle.load(f)
        weights, biases = d["qconv_0_qb"]["weights"]
        for idx in np.ndindex(weights.shape):
            print(idx, weights[idx], parse_nnq(float(weights[idx])))
