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
from amaranth import *
from amaranth.lib import wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out
from amaranth_soc import wishbone
from amaranth.sim import Simulator


# ---------------------------------------------------------------------------
# FakePSRAM — inlined from tiliqua for self-contained test use.
# Copyright (c) 2024 Seb Holzapfel <me@sebholzapfel.com>
# SPDX-License-Identifier: CERN-OHL-S-2.0
# ---------------------------------------------------------------------------


class _FakePSRAM(wiring.Component):
    """Fake PSRAM with configurable latency for simulation."""

    def __init__(
        self, *, addr_width=22, data_width=32, storage_words=512, latency_cycles=4
    ):
        self.latency_cycles = latency_cycles
        self.storage_words = storage_words
        super().__init__(
            {
                "bus": In(
                    wishbone.Signature(
                        addr_width=addr_width,
                        data_width=data_width,
                        granularity=8,
                        features={"cti", "bte"},
                    )
                )
            }
        )

    def elaborate(self, platform):
        m = Module()
        bus = self.bus

        memory = Memory(
            shape=unsigned(self.bus.signature.data_width),
            depth=self.storage_words,
            init=[],
        )
        m.submodules.memory = memory
        mem_wr_port = memory.write_port(granularity=8)
        mem_rd_port = memory.read_port()

        latency_counter = Signal(range(self.latency_cycles + 1))
        in_burst = Signal()
        burst_counter = Signal(8)

        m.d.comb += [
            mem_rd_port.addr.eq(bus.adr),
            mem_wr_port.addr.eq(bus.adr),
            mem_wr_port.data.eq(bus.dat_w),
            bus.dat_r.eq(mem_rd_port.data),
        ]

        m.d.sync += mem_wr_port.en.eq(0)

        with m.FSM():
            with m.State("IDLE"):
                m.d.sync += [
                    bus.ack.eq(0),
                    latency_counter.eq(0),
                    in_burst.eq(0),
                    burst_counter.eq(0),
                ]
                with m.If(bus.cyc & bus.stb):
                    m.d.sync += in_burst.eq(bus.cti != wishbone.CycleType.CLASSIC)
                    m.next = "LATENCY"

            with m.State("LATENCY"):
                m.d.sync += latency_counter.eq(latency_counter + 1)
                with m.If(latency_counter == (self.latency_cycles - 1)):
                    m.next = "RESPOND"

            with m.State("RESPOND"):
                m.d.sync += bus.ack.eq(1)
                with m.If(bus.we):
                    m.d.sync += mem_wr_port.en.eq(bus.sel)
                with m.If(bus.ack):
                    m.d.comb += mem_rd_port.addr.eq(bus.adr + 1)
                with m.If(in_burst):
                    m.d.sync += burst_counter.eq(burst_counter + 1)
                    with m.If(bus.cti == wishbone.CycleType.END_OF_BURST):
                        m.d.sync += bus.ack.eq(0)
                        m.next = "IDLE"
                with m.Else():
                    m.next = "IDLE"

        return m


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

    def _test_activation_cache(self, use_ebr: bool):

        IN_OUT_D = 3
        DILATION_LEVEL = 2
        K = 4
        DILATION = K**DILATION_LEVEL

        fxp_util = util.FxpUtil()
        rng = np.random.default_rng(seed=1337)

        def rnd01(s):
            values = rng.random(size=s)  # (0, 1)
            values = (values * 2) - 1  # (-1, 1)
            values = fxp_util.nparray_to_fixed_point_floats(values)  # snap to closed FP
            return values

        num_samples = 128
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
            in_out_d=IN_OUT_D, dilation_level=DILATION_LEVEL, use_ebr=use_ebr
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

    # def test_activation_cache_ebr(self):
    #     self._test_activation_cache(use_ebr=True)

    # def test_activation_cache_ff(self):
    #     self._test_activation_cache(use_ebr=False)

    def test_network(self):

        if os.getenv("RUN") is None:
            raise Exception("need to set $RUN for weights for test_network")
        if os.getenv("SUB_RUN") is None:
            raise Exception("need to set $SUB_RUN for weights for test_network")

        # TODO: assume finetune for now
        root_dir = (
            Path(__file__).resolve().parents[1]
            / "runs"
            / os.getenv("RUN")
            / os.getenv("SUB_RUN")
        )
        print(">test_network root_dir", root_dir)
        trained_weights = root_dir / "weights/qkeras/latest.pkl"
        layer_info_fname = root_dir / "qkeras_model.layer_info.json"
        test_wave = "sine"
        test_data = root_dir / "test_x_files" / test_wave / "x_yp_yt.pkl"

        with open(layer_info_fname, "r") as f:
            layer_info = json.load(f)

        fxp_model = FxpModel(
            weights_file=str(trained_weights),
            layer_info=layer_info,
            verbose=False,
        )

        dut = QbNetwork.build(str(trained_weights))

        # ---------------------------------------------------------------
        # Wrap dut in a top-level Module and attach one FakePSRAM per
        # ActivationCachePS instance (same pattern as DelayLineTests).
        # Each cache exposes dut.bus_act{i}; storage is sized to cover
        # the full internal 16-bit address space mapped to 32-bit words.
        # ---------------------------------------------------------------
        m = Module()
        m.submodules.dut = dut

        for i, cache in enumerate(dut.activation_caches):
            # internal_addr_width bits → (1<<aw) 16-bit words → half as many 32-bit words
            storage_words = (1 << cache.internal_addr_width) // 2
            psram = _FakePSRAM(
                addr_width=22,
                data_width=32,
                storage_words=max(storage_words, 2),
                latency_cycles=4,
            )
            m.submodules[f"psram_{i}"] = psram
            wiring.connect(m, getattr(dut, f"bus_act{i}"), psram.bus)
        # ---------------------------------------------------------------

        with open(test_data, "rb") as f:
            data = pickle.load(f)
            x = np.asarray(data["x"])

        x = x.reshape(-1, x.shape[-1])
        self.assertEqual(x.shape[1], dut.IN_D)

        y_pred_fxp_cache_fname = (
            root_dir / "test_x_files" / test_wave / "test_network.y_pred_fxp.pkl"
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
            ctx.set(dut.o.ready, 1)

            for sample in tqdm.tqdm(x, desc="amaranth"):
                ctx.set(dut.i.payload, parse_nnq(sample, assert_exact=False))
                ctx.set(dut.i.valid, 1)
                await ctx.tick()

                for _ in range(10000):
                    if ctx.get(dut.o.valid):
                        break
                    await ctx.tick()
                assert ctx.get(dut.o.valid)

                y_pred_am.append(ctx.get(dut.o.payload).as_float())

        sim = Simulator(m)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()

        df = pd.DataFrame()
        df["x"] = x[:, 0]  # just phase_sin
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
            plt_fname = f"{root_dir}/amaranth.{test_wave}.jittered.png"
            print("saving plot to", plt_fname)
            plt.savefig(plt_fname)
            plt.clf()

        np.testing.assert_allclose(y_pred_fxp, y_pred_am, atol=0.005)
