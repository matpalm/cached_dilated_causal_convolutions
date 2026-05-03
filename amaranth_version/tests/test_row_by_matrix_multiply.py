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

        weights = np.array(
            [
                [-0.759765625, 0.599853515625, -0.158935546875, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [-0.759765625, 0.599853515625, -0.158935546875, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        dut = RowByMatrixMultiply(weights)

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            inp = parse_nnq([0.0, 0.0625, 0.0, 0.0])

            ctx.set(dut.i.payload, inp)
            ctx.set(dut.i.valid, 1)
            await ctx.tick()
            ctx.set(dut.i.valid, 0)

            for _ in range(dut.IN_D + 4):
                if ctx.get(dut.o.valid):
                    break
                await ctx.tick()

            self.assertEqual(ctx.get(dut.o.valid), 1)

            expected = []
            for col in weights:
                expected.append(
                    sum(
                        a.as_float() * b.as_float() for a, b in zip(inp, parse_nnq(col))
                    )
                )
            for j, expected_val in enumerate(expected):
                actual = ctx.get(dut.o.payload[j]).as_float()
                self.assertAlmostEqual(actual, expected_val)

            await ctx.tick()
            self.assertEqual(ctx.get(dut.o.valid), 0)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()
