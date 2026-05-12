from pathlib import Path
import sys
import unittest
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import os

# for amaranth_future :/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "amaranth_version" / "src"))

from amaranth_version.src.cdcc import conv1d

from fxpmath_version import fxpmath_conv1d_qb
from fxpmath_version import util
from fxpmath_version.fxpmath_model import FxpModel
from fxpmath_version.activation_cache import ActivationCache as FxpActivationCache

from cdcc import parse_nnq, NNQ, NNQ_DW
from cdcc.activation_cache import ActivationCache as AmaranthActivationCache
from cdcc.dot_product import DotProduct
from cdcc.row_by_matrix_multiply import RowByMatrixMultiply
from cdcc.conv1d import Conv1d
from cdcc.qb_network import QbNetwork
from amaranth.sim import Simulator

# cd ~/dev/cached_dilated_causal_convolutions
# python -m unittest discover test_equivalences/ -k TestDotProductEquivalence

class TestEquivalences(unittest.TestCase):

    def test_dot_product(self):

        D = 5

        # make some random data
        fxp_util = util.FxpUtil()
        rng = np.random.default_rng(seed=1337)
        def rnd01(s):
            values = rng.random(size=s)  # (0, 1)
            values = ( values * 2 ) - 1  # (-1, 1)
            values = fxp_util.nparray_to_fixed_point_floats(values)  # snap to closed FP
            return values

        eg_inp = rnd01((D,))
        weights = rnd01((D,))

        # run fxp version
        # note: have to make fake conv, just to use dot_product method :/
        K = 4
        dummy_weights = np.zeros((K, D, D))
        dummy_biases = np.zeros((D))
        dummy_conv = fxpmath_conv1d_qb.FxpMathConv1DQuantisedBitsBlock(
            fxp_util, "layer_name", dummy_weights, dummy_biases, verbose=False
        )
        accum = fxp_util.double_width(0)
        dummy_conv.dot_product(eg_inp, weights, accum)
        fxp_math_result = float(accum)

        # run amaranth version
        dut = DotProduct(weights)

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            inp = parse_nnq(eg_inp)
            ctx.set(dut.i.payload, inp)
            ctx.set(dut.i.valid, 1)

            for _ in range(100):
                if ctx.get(dut.o.valid):
                    break
                await ctx.tick()

            self.assertEqual(ctx.get(dut.o.valid), 1)

            amaranth_result = ctx.get(dut.o.payload).as_float()

            self.assertAlmostEqual(
                fxp_math_result, amaranth_result)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()

    def test_row_by_matrix_multiply(self):

        K, IN_D, OUT_D = 4, 2, 3

        # make some random data
        fxp_util = util.FxpUtil()
        rng = np.random.default_rng(seed=1337)
        def rnd01(s):
            values = rng.random(size=s)  # (0, 1)
            values = ( values * 2 ) - 1  # (-1, 1)
            values = fxp_util.nparray_to_fixed_point_floats(values)  # snap to closed FP
            return values

        eg_inp = rnd01((IN_D,))
        row_wise_weights = rnd01((IN_D, OUT_D))

        # run fxp version
        # note: have to make fake conv, just to use row_by_matrix_multiply method :/
        dummy_weights = np.zeros((K, IN_D, OUT_D))
        dummy_biases = np.zeros((OUT_D))
        dummy_conv = fxpmath_conv1d_qb.FxpMathConv1DQuantisedBitsBlock(
            fxp_util, "layer_name", dummy_weights, dummy_biases, verbose=False
        )
        accums = [fxp_util.double_width(0) for _ in range(OUT_D)]
        col_wise_weights = row_wise_weights.T
        dummy_conv.row_by_matrix_multiply(eg_inp, col_wise_weights, accums)
        fxp_result = list(map(float, accums))

        # run amaranth version
        dut = RowByMatrixMultiply(row_wise_weights)

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            inp = parse_nnq(eg_inp)
            ctx.set(dut.i.payload, inp)
            ctx.set(dut.i.valid, 1)

            for _ in range(100):
                if ctx.get(dut.o.valid):
                    break
                await ctx.tick()

            amaranth_result = ctx.get(dut.o.payload)
            amaranth_result = [v.as_float() for v in amaranth_result]

            np.testing.assert_allclose(fxp_result, amaranth_result)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()

    def test_conv1d(self):

        K, IN_D, OUT_D = 4, 2, 3

        # make some random data
        fxp_util = util.FxpUtil()
        rng = np.random.default_rng(seed=1337)

        def rnd01(s):
            values = rng.random(size=s)  # (0, 1)
            values = (values * 2) - 1  # (-1, 1)
            values = fxp_util.nparray_to_fixed_point_floats(values)  # snap to closed FP
            return values

        eg_inp = rnd01((K, IN_D))
        weights = rnd01((K, IN_D, OUT_D))

        # biases = rnd01((OUT_D,))
        biases = fxp_util.nparray_to_fixed_point_floats(
            np.zeros(
                OUT_D,
            )
        )

        # run fxp version
        fxp_conv = fxpmath_conv1d_qb.FxpMathConv1DQuantisedBitsBlock(
            fxp_util, "layer_name", weights, biases, verbose=False
        )
        fxp_result = fxp_conv.apply(eg_inp)

        # run amaranth version
        dut = Conv1d(weights, biases, apply_relu=False)

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            inp = parse_nnq(eg_inp)
            ctx.set(dut.i.payload, inp)
            ctx.set(dut.i.valid, 1)

            for _ in range(100):
                if ctx.get(dut.o.valid):
                    break
                await ctx.tick()

            amaranth_result = ctx.get(dut.o.payload)
            amaranth_result = [v.as_float() for v in amaranth_result]

            np.testing.assert_allclose(fxp_result, amaranth_result)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()

    def test_activation_cache(self):

        IN_OUT_D = 3
        DILATION_LEVEL = 1
        K = 4
        DILATION = K**DILATION_LEVEL

        fxp_util = util.FxpUtil()
        rng = np.random.default_rng(seed=1337)

        def rnd01(s):
            values = rng.random(size=s)  # (0, 1)
            values = (values * 2) - 1  # (-1, 1)
            values = fxp_util.nparray_to_fixed_point_floats(values)  # snap to closed FP
            return values

        num_samples = 32
        samples = rnd01((num_samples, IN_OUT_D))

        fxp_cache = FxpActivationCache(
            depth=IN_OUT_D,
            dilation=DILATION,
            kernel_size=K,
        )
        fxp_results = []
        for i, sample in enumerate(samples):
            fxp_result = fxp_cache.apply(sample)
            print("fxp_result", i, fxp_result)
            fxp_results.append(fxp_result)

        dut = AmaranthActivationCache(
            in_out_d=IN_OUT_D,
            dilation_level=DILATION_LEVEL,
        )
        amaranth_results = []

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            for i, sample in enumerate(samples):
                ctx.set(dut.i.payload, parse_nnq(sample, assert_exact=False))
                ctx.set(dut.i.valid, 1)
                await ctx.tick()

                for _ in range(100):
                    if ctx.get(dut.o.valid):
                        break
                    await ctx.tick()
                assert ctx.get(dut.o.valid)

                am_output = ctx.get(dut.o.payload)
                am_output = [[v.as_float() for v in row] for row in am_output]
                print("am_output", i, am_output)
                amaranth_results.append(np.asarray(am_output))

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()

        self.assertEqual(len(fxp_results), len(amaranth_results))
        for i in range(len(fxp_results)):
            np.testing.assert_allclose(
                np.asarray(fxp_results[i]),
                np.asarray(amaranth_results[i]),
                err_msg=f"failed at i={i}",
            )

    def test_network(self):

        if os.getenv("RUN") is None:
            raise Exception("need to set $RUN for weights for test_network")

        root_dir = Path(__file__).resolve().parents[1] / "runs" / os.getenv("RUN")
        print(">test_network root_dir", root_dir)
        trained_weights = root_dir / "weights/qkeras/latest.pkl"
        layer_info_fname = root_dir / "qkeras_model.layer_info.json"
        test_data = root_dir / "test_x_files/zigzag/x_yp_yt.pkl"

        with open(layer_info_fname, "r") as f:
            layer_info = json.load(f)

        fxp_model = FxpModel(
            weights_file=str(trained_weights),
            layer_info=layer_info,
            verbose=False,
        )

        dut = QbNetwork.build(str(trained_weights))

        with open(test_data, "rb") as f:
            data = pickle.load(f)
            x = np.asarray(data["x"])

        x = x.reshape(-1, x.shape[-1])
        self.assertEqual(x.shape[1], dut.IN_D)

        # print("HACK: 50 x samples")
        # x = x[:50]

        y_pred_fxp_cache_fname = (
            root_dir / "test_x_files/zigzag/test_network.y_pred_fxp.pkl"
        )
        if os.path.exists(y_pred_fxp_cache_fname):
            print("using cached", y_pred_fxp_cache_fname)
            with open(y_pred_fxp_cache_fname, "rb") as f:
                y_pred_fxp = pickle.load(f)
            if len(y_pred_fxp) != len(x):
                raise Exception(
                    f"cache invalid; |x|={len(x)} but cached |y_pred_fxp|={len(y_pred_fxp)}"
                )
        else:
            y_pred_fxp = []
            for sample in tqdm.tqdm(x, desc="fxpmath"):
                y_pred_fxp.append(float(fxp_model.predict(sample)[0]))
            with open(y_pred_fxp_cache_fname, "wb") as f:
                pickle.dump(y_pred_fxp, f)

        y_pred_am = []

        async def testbench(ctx):
            ctx.set(dut.i.valid, 0)
            ctx.set(dut.o.ready, 1)

            for sample in tqdm.tqdm(x, desc="amaranth"):
                while not ctx.get(dut.i.ready):
                    await ctx.tick()

                raise Exception(
                    "need to check this. tests show correctness, but this is faulty?"
                )

                ctx.set(dut.i.payload, parse_nnq(sample, assert_exact=False))
                ctx.set(dut.i.valid, 1)
                await ctx.tick()
                # ctx.set(dut.i.valid, 0)  TOOD: we don't want this?

                for _ in range(10000):
                    if ctx.get(dut.o.valid):
                        break
                    await ctx.tick()

                y_pred_am.append(ctx.get(dut.o.payload).as_float())

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()

        df = pd.DataFrame()
        df["x"] = x[:, 0]
        df["y_pred_fxp"] = y_pred_fxp
        df["y_pred_am"] = y_pred_am
        df["n"] = range(len(x))

        # jitter the plotted versions
        scale = 0.05
        jitter = np.random.default_rng(seed=1337).uniform(-scale, scale, size=len(df))
        df["y_pred_fxp_jittered"] = df["y_pred_fxp"] + jitter
        df["y_pred_am_jittered"] = df["y_pred_am"] - jitter

        wide_df = pd.melt(
            df,
            id_vars=["n"],
            value_vars=["x", "y_pred_fxp_jittered", "y_pred_am_jittered"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            p = sns.lineplot(wide_df, x="n", y="value", hue="variable")
            p.set(ylim=(-2, 2))
            plt_fname = f"{root_dir}/amaranth.zigzag.png"
            print("saving plot to", plt_fname)
            plt.savefig(plt_fname)
            plt.clf()

        np.testing.assert_allclose(y_pred_fxp, y_pred_am, atol=0.005)
