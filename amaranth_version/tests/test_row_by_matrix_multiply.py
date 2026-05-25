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

    def _run_row_by_matrix_multiply_single_vector(self, in_d, out_d):

        # Seeded random generation keeps tests deterministic.
        # Use quarter-step values so they are quantization-friendly.
        rng = np.random.default_rng(1234 + (in_d * 1000) + out_d)
        weights = rng.integers(-8, 9, size=(in_d, out_d)).astype(float) / 4.0

        dut = RowByMatrixMultiply(weights)

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            inp_float = (rng.integers(-8, 9, size=in_d).astype(float) / 4.0).tolist()
            inp = parse_nnq(inp_float)

            ctx.set(dut.i.payload, inp)
            ctx.set(dut.i.valid, 1)
            await ctx.tick()
            ctx.set(dut.i.valid, 0)

            for _ in range(1000):
                if ctx.get(dut.o.valid):
                    break
                await ctx.tick()
            self.assertEqual(ctx.get(dut.o.valid), 1)

            expected_float = np.array(inp_float) @ weights
            expected = parse_nnq(expected_float.tolist())
            for j, expected_val in enumerate(expected):
                actual_val = ctx.get(dut.o.payload[j])
                self.assertAlmostEqual(actual_val.as_float(), expected_val.as_float())

            await ctx.tick()
            self.assertEqual(ctx.get(dut.o.valid), 0)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()

    def test_row_by_matrix_multiply_single_vector_4_8(self):
        self._run_row_by_matrix_multiply_single_vector(in_d=4, out_d=8)

    def test_row_by_matrix_multiply_single_vector_4_12(self):
        self._run_row_by_matrix_multiply_single_vector(in_d=4, out_d=12)

    def test_row_by_matrix_multiply_single_vector_4_16(self):
        self._run_row_by_matrix_multiply_single_vector(in_d=4, out_d=16)
