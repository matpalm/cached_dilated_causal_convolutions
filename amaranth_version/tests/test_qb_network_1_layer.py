from pathlib import Path
import sys
import unittest
import pickle
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amaranth.sim import Simulator

from cdcc import parse_nnq
from cdcc.qb_network_1_layer import QbNetworkOneLayer


class TestQbNetworkOneLayer(unittest.TestCase):

    def test_fxp_math_equiv(self):
        trained_weights = "runs/41_tiliqua_1layer/weights/qkeras/latest.pkl"
        test_data = "runs/41_tiliqua_1layer/test_x_files/sine/x_yp_yt.pkl"

        dut = QbNetworkOneLayer.build(trained_weights)

        with open(test_data, "rb") as f:
            data = pickle.load(f)
            x = np.asarray(data["x"])
            y_true = np.asarray(data["y_true"])
            y_pred_fxp = np.asarray(data["y_pred"])

        x = x.reshape(-1, x.shape[-1])
        self.assertEqual(x.shape[1], dut.IN_D)

        y_pred_am = []

        async def testbench(ctx):
            ctx.set(dut.i.valid, 0)
            ctx.set(dut.o.ready, 1)

            for sample in x:
                while not ctx.get(dut.i.ready):
                    if ctx.get(dut.o.valid):
                        y_pred_am.append(ctx.get(dut.o.payload).as_float())
                    await ctx.tick()

                ctx.set(dut.i.payload, parse_nnq(sample, assert_exact=False))
                ctx.set(dut.i.valid, 1)

                if ctx.get(dut.o.valid):
                    y_pred_am.append(ctx.get(dut.o.payload).as_float())

                await ctx.tick()
                ctx.set(dut.i.valid, 0)

                if ctx.get(dut.o.valid):
                    y_pred_am.append(ctx.get(dut.o.payload).as_float())

            max_drain_cycles = len(x) * 256
            for _ in range(max_drain_cycles):
                if len(y_pred_am) >= len(x):
                    break
                if ctx.get(dut.o.valid):
                    y_pred_am.append(ctx.get(dut.o.payload).as_float())
                await ctx.tick()

            # self.assertGreater(len(outputs), 0)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()

        # save plot
        df = pd.DataFrame()
        # plot just waveform for x and ignore e0 and e1
        df["x"] = x[:, 0]
        # recall: y_true, and fxpmath version, use simple old code version
        #         where output is always 4d
        # recall: fxp math replicated simple y_true 4d ( simple code )
        df["y_true"] = y_true[:, 0]
        df["y_pred_fxp"] = y_pred_fxp[:, 0]
        df["y_pred_am"] = y_pred_am
        df["n"] = range(len(x))
        wide_df = pd.melt(
            df, id_vars=["n"], value_vars=["x", "y_true", "y_pred_fxp", "y_pred_am"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            p = sns.lineplot(wide_df, x="n", y="value", hue="variable")
            p.set(ylim=(-2, 2))
            plt_fname = "foo.png"  # f"{opts.plot_dir}/fxp_math.y_pred.{wave}.png"

            plt.savefig(plt_fname)
            plt.clf()
