from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaranth.sim import Simulator

from cdcc import parse_nnq
from cdcc.conv1d import Conv1d


class TestConv1d(unittest.TestCase):

    def _general_test(self, weights, biases, apply_relu, inputs, expected):

        dut = Conv1d(weights, biases, apply_relu)

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            ctx.set(dut.i.payload, inputs)
            ctx.set(dut.i.valid, 1)
            await ctx.tick()
            ctx.set(dut.i.valid, 0)

            for _ in range(dut.IN_D + 8):
                if ctx.get(dut.o.valid):
                    break
                await ctx.tick()

            self.assertEqual(ctx.get(dut.o.valid), 1)

            for j, expected_val in enumerate(expected):
                actual = ctx.get(dut.o.payload[j]).as_float()
                self.assertAlmostEqual(actual, expected_val.as_float())

            await ctx.tick()
            self.assertEqual(ctx.get(dut.o.valid), 0)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()

    def test_conv1d_with_only_one_non_zero_set_of_weights(self):
        # should be the same as the test_row_by_multiply

        # weights for a in_d=2 out_d=3 mult
        K, IN_D, OUT_D = 4, 2, 3
        weights = np.zeros((K, IN_D, OUT_D))
        weights[0] = np.array([[-2, 1, 0], [0.5, -1, 0.25]])
        print("weights", weights)

        biases = np.array([0, 0, 0])
        print("biases", biases)

        inputs = [
            parse_nnq([0.5, -1.0]),
            parse_nnq([0.5, -1.0]),
            parse_nnq([0.5, -1.0]),
            parse_nnq([0.5, -1.0]),
        ]
        print("inputs", inputs)

        expected = parse_nnq([-1.5, 1.5, -0.25])

        apply_relu = True
        self._general_test(weights, biases, not apply_relu, inputs, expected)

    def test_conv1d_with_pos_neg_ones_weights(self):
        # should be the same as the test_row_by_multiply

        # weights for a in_d=2 out_d=3 mult
        K, IN_D, OUT_D = 4, 2, 3
        weights = np.empty((K, IN_D, OUT_D))
        weights[0] = np.ones((IN_D, OUT_D))
        weights[1] = -np.ones((IN_D, OUT_D))
        weights[2] = np.ones((IN_D, OUT_D))
        weights[3] = -np.ones((IN_D, OUT_D))
        print("weights", weights)

        biases = np.array([0, 0, 0])
        print("biases", biases)

        inputs = [
            parse_nnq([0.5, -1.0]),
            parse_nnq([0.5, -1.0]),
            parse_nnq([0.5, -1.0]),
            parse_nnq([0.5, -1.0]),
        ]
        print("inputs", inputs)

        expected = parse_nnq([0, 0, 0])

        apply_relu = True
        self._general_test(weights, biases, not apply_relu, inputs, expected)

    def test_conv1d_with_pos_neg_ones_weights_with_bias(self):
        # should be the same as the test_row_by_multiply

        # weights for a in_d=2 out_d=3 mult
        K, IN_D, OUT_D = 4, 2, 3
        weights = np.empty((K, IN_D, OUT_D))
        weights[0] = np.ones((IN_D, OUT_D))
        weights[1] = -np.ones((IN_D, OUT_D))
        weights[2] = np.ones((IN_D, OUT_D))
        weights[3] = -np.ones((IN_D, OUT_D))
        print("weights", weights)

        biases = np.array([0.25, -0.125, 0])
        print("biases", biases)

        inputs = [
            parse_nnq([0.5, -1.0]),
            parse_nnq([0.5, -1.0]),
            parse_nnq([0.5, -1.0]),
            parse_nnq([0.5, -1.0]),
        ]
        print("inputs", inputs)

        expected = parse_nnq([0.25, -0.125, 0])

        apply_relu = True
        self._general_test(weights, biases, not apply_relu, inputs, expected)

    # def test_conv1d_out_d_one(self):

    #     N_KERNELS, OUT_D, IN_D = 4, 1, 5

    #     weights = np.zeros((N_KERNELS, OUT_D, IN_D))
    #     weights[0, 0, 0] = 0.25
    #     weights[0, 0, 1] = -0.5
    #     biases = np.array([0])  # 0.5])

    #     dut = Conv1d(weights, biases, apply_relu=False)

    #     async def testbench(ctx):
    #         ctx.set(dut.o.ready, 1)

    #         inp = [
    #             parse_nnq([2.0, -1.0, 0.0, 0.0, 0.0]),
    #             parse_nnq([0.0, 0.0, 0.0, 0.0, 0.0]),
    #             parse_nnq([0.0, 0.0, 0.0, 0.0, 0.0]),
    #             parse_nnq([0.0, 0.0, 0.0, 0.0, 0.0]),
    #         ]

    #         ctx.set(dut.i.payload, inp)
    #         ctx.set(dut.i.valid, 1)
    #         await ctx.tick()
    #         ctx.set(dut.i.valid, 0)

    #         for _ in range(dut.IN_D + 8):
    #             if ctx.get(dut.o.valid):
    #                 break
    #             await ctx.tick()

    #         self.assertEqual(ctx.get(dut.o.valid), 1)

    #         actual = ctx.get(dut.o.payload[0]).as_float()
    #         self.assertAlmostEqual(actual, 1.0)

    #         await ctx.tick()
    #         self.assertEqual(ctx.get(dut.o.valid), 0)

    #     sim = Simulator(dut)
    #     sim.add_clock(1e-6, domain="sync")
    #     sim.add_testbench(testbench)
    #     sim.run()

    # def test_conv1d_without_relu(self):

    #     N_KERNELS, OUT_D, IN_D = 4, 3, 5

    #     weights = np.zeros((N_KERNELS, OUT_D, IN_D))
    #     weights[0, 0, 0] = 1.0
    #     weights[0, 1, 1] = -0.5
    #     weights[0, 2, 2] = -0.25
    #     biases = np.array([0.25, -0.125, 0.5])

    #     dut = Conv1d(weights, biases, apply_relu=False)

    #     async def testbench(ctx):
    #         ctx.set(dut.o.ready, 1)

    #         inp = [
    #             parse_nnq([0.5, 1.0, -2.0, 0.0, 0.0]),
    #             parse_nnq([3.0, -1.0, 0.25, 0.0, 0.0]),
    #             parse_nnq([0.0, 0.0, 0.0, 0.0, 0.0]),
    #             parse_nnq([0.0, 0.0, 0.0, 0.0, 0.0]),
    #         ]

    #         ctx.set(dut.i.payload, inp)
    #         ctx.set(dut.i.valid, 1)
    #         await ctx.tick()
    #         ctx.set(dut.i.valid, 0)

    #         for _ in range(dut.IN_D + 8):
    #             if ctx.get(dut.o.valid):
    #                 break
    #             await ctx.tick()

    #         self.assertEqual(ctx.get(dut.o.valid), 1)

    #         # expected = [0.5, 0.0, 0.5]  # no bias
    #         expected = [0.75, -0.625, 1.0]
    #         for j, expected_val in enumerate(expected):
    #             actual = ctx.get(dut.o.payload[j]).as_float()
    #             self.assertAlmostEqual(actual, expected_val)

    #         await ctx.tick()
    #         self.assertEqual(ctx.get(dut.o.valid), 0)

    #     sim = Simulator(dut)
    #     sim.add_clock(1e-6, domain="sync")
    #     sim.add_testbench(testbench)
    #     sim.run()
