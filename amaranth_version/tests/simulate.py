"""
uv run python -m amaranth_version.tests.simulate \
    --run 263_pc600_rnd_flip_no_skip --sub-run finetune \
    --a-cv 0.6 --b-cv 0.6 --morph-cv 1.0
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from amaranth import Module
from amaranth.lib import wiring
from amaranth.sim import Simulator

from cdcc import parse_nnq
from cdcc import K
from cdcc.qb_network import QbNetwork

from fake_psram import FakePSRAM

from common.synthetic_data import build_triangle_sample

def _wrap_with_psram(dut):
    """
    Following test_network, wrap the dut in a top module with one FakePSRAM
    per PSRAM backed activation cache.
    """
    m = Module()
    m.submodules.dut = dut

    # connect a FakePSRAM to each PSRAM backed cache ( if any ).
    for i in dut.psram_cache_indices:
        cache = dut.activation_caches[i]
        # internal_addr_width bits -> (1<<aw) 16-bit words -> half as many 32-bit words
        storage_words = (1 << cache.internal_addr_width) // 2
        psram = FakePSRAM(
            addr_width=22,
            data_width=32,
            storage_words=max(storage_words, 2),
            latency_cycles=4,
        )
        m.submodules[f"psram_{i}"] = psram
        wiring.connect(m, getattr(dut, f"bus_act{i}"), psram.bus)

    return m


def _run(dut, top, samples, max_wait=20000):
    """Stream samples through the network, one per input handshake."""
    y_pred = []

    async def testbench(ctx):
        ctx.set(dut.o.ready, 1)

        for sample in samples:
            ctx.set(dut.i.payload, parse_nnq(list(sample), assert_exact=False))
            ctx.set(dut.i.valid, 1)
            await ctx.tick()

            produced = False
            for _ in range(max_wait):
                if ctx.get(dut.o.valid):
                    produced = True
                    break
                await ctx.tick()
            if not produced:
                raise RuntimeError("network produced no output for a sample")

            y_pred.append(ctx.get(dut.o.payload).as_float())

    sim = Simulator(top)
    sim.add_clock(1e-6, domain="sync")
    sim.add_testbench(testbench)
    sim.run()
    return y_pred


def simulate(
    run: str,
    sub_run: str,
    a_cv: float,
    b_cv: float,
    morph_cv: float,
    test_seq_len: int,
    tri_freq: float,
    output_plot: str,
):

    assert -1 <= a_cv <= 1
    assert -1 <= b_cv <= 1
    assert -1 <= morph_cv <= 1

    # note: this file lives in amaranth_version/tests/ so runs/ is at parents[2]
    root_dir = Path(__file__).resolve().parents[2] / "runs" / run / sub_run
    trained_weights = root_dir / "weights" / "qkeras" / "latest.pkl"
    if not trained_weights.exists():
        raise FileNotFoundError(f"no trained weights at {trained_weights}")

    # build the real network directly from this run's quantised weights
    # ( QbNetwork handles the qconv_regressor_qb final layer ).
    dut = QbNetwork.build(str(trained_weights))
    top = _wrap_with_psram(dut)

    # drive the real network with a synthetic sample built like
    # test.build_base_x: a triangle core wave in col 0 plus fixed
    # a_cv / b_cv / morph_cv columns.
    #
    # the network needs to warm up, so ( as in qkeras_version/test.py ) we
    # prepend receptive_field_size samples before the test_seq_len we care
    # about. receptive_field_size == K**num_dilated_layers, where the final
    # ( regressor ) layer is not a dilated layer.
    num_dilated_layers = dut.num_layers - 1
    receptive_field_size = K**num_dilated_layers
    seq_len_plus_rf = receptive_field_size + test_seq_len
    samples = build_triangle_sample(
        seq_len=seq_len_plus_rf,
        a_cv=a_cv,
        b_cv=b_cv,
        morph_cv=morph_cv,
        tri_freq=tri_freq,
    )

    y_pred = _run(dut, top, samples)

    # plot input ch0 vs y_pred, equivalent to qkeras_version/test.py.
    n = np.arange(seq_len_plus_rf)
    tri = samples[:, 0]
    y_pred = np.asarray(y_pred, dtype=np.float32)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(n, tri, linewidth=1.5, label="input ch0")
    ax.plot(n, y_pred, linewidth=1.5, label="y_pred")
    ax.axvline(
        receptive_field_size,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"rf={receptive_field_size}",
    )
    ax.set_ylabel("value")
    ax.set_title(
        f"triangle={tri_freq:.2f}Hz, a_cv={a_cv}, b_cv={b_cv}, morph_cv={morph_cv}"
    )
    ax.set_xlabel("sample index")
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(output_plot)
    plt.close(fig)
    print("saved plot", output_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run", required=True, help="run name under runs/")
    parser.add_argument(
        "--sub-run", required=True, help="sub run ( e.g. pretrain / finetune )"
    )
    parser.add_argument("--a-cv", type=float, required=True)
    parser.add_argument("--b-cv", type=float, required=True)
    parser.add_argument("--morph-cv", type=float, required=True)
    parser.add_argument("--test-seq-len", type=int, default=400)
    parser.add_argument("--tri-freq", type=float, default=300.0)
    parser.add_argument("--output-plot", type=str, required=True)
    opts = parser.parse_args()

    simulate(
        run=opts.run,
        sub_run=opts.sub_run,
        a_cv=opts.a_cv,
        b_cv=opts.b_cv,
        morph_cv=opts.morph_cv,
        test_seq_len=opts.test_seq_len,
        tri_freq=opts.tri_freq,
        output_plot=opts.output_plot,
    )
