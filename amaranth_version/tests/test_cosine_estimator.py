from pathlib import Path
import sys
import unittest
import math
import matplotlib.pyplot as plt
import seaborn as sns

from amaranth_future import fixed
from amaranth.sim import Simulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdcc import NNQ
from cdcc.cosine_estimator import CosineEstimator


class TestCosineEstimator(unittest.TestCase):

    def test_cosine_estimator(self):
        dut = CosineEstimator()
        freq_hz = 440.0
        sample_rate_hz = 192_000.0
        samples_for_two_cycles = int(math.ceil(2.0 * sample_rate_hz / freq_hz))
        total_cycles = (samples_for_two_cycles + 256) * 10

        input_samples = []
        true_cos_samples = []
        output_cycles = []
        derived_cos_samples = []

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            for i in range(total_cycles):
                sample = math.sin(2.0 * math.pi * freq_hz * i / sample_rate_hz)
                input_samples.append(sample)

                true_cos_samples.append(
                    math.cos(2.0 * math.pi * freq_hz * i / sample_rate_hz)
                )

                ctx.set(dut.i.payload, fixed.Const(sample, shape=NNQ))
                ctx.set(dut.i.valid, 1)
                await ctx.tick()

                if ctx.get(dut.o.valid):
                    derived_cos = ctx.get(dut.o.payload).as_float()
                    output_cycles.append(i)
                    derived_cos_samples.append(derived_cos)

        sim = Simulator(dut)
        sim.add_clock(1e-6, domain="sync")
        sim.add_testbench(testbench)
        sim.run()

        window_start = max(0, total_cycles - samples_for_two_cycles)
        window_input_x = list(range(window_start, total_cycles))
        window_input = input_samples[window_start:total_cycles]
        window_true_cos = true_cos_samples[window_start:total_cycles]

        window_output_pairs = [
            (cyc, val)
            for cyc, val in zip(output_cycles, derived_cos_samples)
            if cyc >= window_start
        ]
        window_output_x = [cyc for cyc, _ in window_output_pairs]
        window_derived = [val for _, val in window_output_pairs]

        plot_x = window_input_x + window_input_x + window_output_x
        plot_y = window_input + window_true_cos + window_derived
        plot_label = (
            ["sin"] * len(window_input)
            + ["true cos"] * len(window_true_cos)
            + ["derived cos"] * len(window_derived)
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        sns.lineplot(x=plot_x, y=plot_y, hue=plot_label, ax=ax)
        ax.set_title("sin vs y_true cos vs y_pred cos (last two cycles)")
        ax.set_xlabel("sample")
        ax.set_ylabel("amplitude")
        plt.tight_layout()
        plt.savefig("test_cosine_injection_waveforms.png")
        plt.close()
