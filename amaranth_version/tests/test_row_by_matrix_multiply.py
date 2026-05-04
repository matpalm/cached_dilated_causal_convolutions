from pathlib import Path
import sys
import unittest
import json
import tempfile

from amaranth.sim import Simulator

import numpy as np

from cdcc import parse_nnq
from cdcc.row_by_matrix_multiply import RowByMatrixMultiply


class TestRowByMatrixMultiply(unittest.TestCase):

    def test_row_by_matrix_multiply_single_vector(self):

        # weights for a in_d=2 out_d=3 mult
        IN_D, OUT_D = 2, 3
        weights = np.array([[-2, 1, 0], [0.5, -1, 0.25]])
        assert weights.shape == (IN_D, OUT_D)

        dut = RowByMatrixMultiply(weights)

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            # select first column / 2 - second column
            inp = parse_nnq([0.5, -1.0])

            ctx.set(dut.i.payload, inp)
            ctx.set(dut.i.valid, 1)
            await ctx.tick()
            ctx.set(dut.i.valid, 0)

            for _ in range(dut.IN_D + 4):
                if ctx.get(dut.o.valid):
                    break
                await ctx.tick()
            self.assertEqual(ctx.get(dut.o.valid), 1)

            expected = parse_nnq([-1.5, 1.5, -0.25])
            for j, expected_val in enumerate(expected):
                actual_val = ctx.get(dut.o.payload[j])
                self.assertAlmostEqual(actual_val.as_float(), expected_val.as_float())

            await ctx.tick()
            self.assertEqual(ctx.get(dut.o.valid), 0)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()
