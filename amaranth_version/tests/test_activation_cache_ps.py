"""
Tests for ActivationCachePS — PSRAM-backed activation cache.

Test structure combines patterns from:
  - TestActivationCache  (test_activation_cache.py)
  - DelayLineTests       (tiliqua/gateware/tests/test_delayln.py)

FakePSRAM and stream helpers are inlined from tiliqua so this file is
self-contained without depending on the tiliqua package.

Original tiliqua copyright: (c) 2024 Seb Holzapfel <me@sebholzapfel.com>
Original SPDX-License-Identifier: CERN-OHL-S-2.0
"""

from pathlib import Path
import sys
import unittest
import numpy as np

from amaranth import *
from amaranth.lib import stream, wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out
from amaranth.sim import Simulator
from amaranth_soc import wishbone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdcc import parse_nnq, NNQ, K
from cdcc.activation_cache_ps import ActivationCachePS

# ============================================================
# Inlined from tiliqua/gateware/src/tiliqua/test/psram.py
# Copyright (c) 2024 Seb Holzapfel <me@sebholzapfel.com>
# SPDX-License-Identifier: CERN-OHL-S-2.0
# ============================================================


class FakePSRAM(wiring.Component):
    """
    Fake PSRAM for testbenches. Simulates classic and burst transactions with
    configurable access latency, matching real PSRAM behaviour.
    """

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
                    is_burst = bus.cti != wishbone.CycleType.CLASSIC
                    m.d.sync += in_burst.eq(is_burst)
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
                    end_of_burst = bus.cti == wishbone.CycleType.END_OF_BURST
                    with m.If(end_of_burst):
                        m.d.sync += bus.ack.eq(0)
                        m.next = "IDLE"
                with m.Else():
                    m.next = "IDLE"

        return m


# ============================================================
# Inlined from tiliqua/gateware/src/tiliqua/test/stream.py
# Original author: Vegard Storheil Eriksen <zyp@jvnv.net>  (MIT)
# ============================================================


async def _stream_put(ctx, s, payload):
    ctx.set(s.valid, 1)
    ctx.set(s.payload, payload)
    await ctx.tick().until(s.ready == 1)
    ctx.set(s.valid, 0)


# ============================================================
# Test class
# ============================================================


class TestActivationCachePS(unittest.TestCase):
    """
    Verifies ActivationCachePS produces the same delayed-sample outputs as
    the EBR-backed ActivationCache, using PSRAM (FakePSRAM) as backing store.

    Address space calculation for the default parameters
    (in_out_d=3, dilation_level=1, burst_len=4, cachesize_words=64):
      num_entries   = 4^2 = 16
      internal words= 3 * 16 = 48  (16-bit NNQ words)
      external words= 24  (32-bit, two NNQ per word via _WishboneAdapter)
      FakePSRAM storage_words = 32  (next power-of-2 >= 24)
    """

    def _run_activation_cache_ps(
        self, in_out_d, dilation_level, cache_kwargs=None, n_steps=20, max_wait=2000
    ):
        dut = ActivationCachePS(
            in_out_d=in_out_d,
            dilation_level=dilation_level,
            addr_width_o=22,
            base=0,
            cache_kwargs=cache_kwargs,
        )

        # External word count: ceil(in_out_d * num_entries / 2) + margin
        import math

        ext_words = math.ceil(in_out_d * dut.num_entries / 2)
        # Round up to next power-of-two for FakePSRAM
        storage_words = 1 << math.ceil(math.log2(ext_words + 1)) if ext_words > 0 else 2

        m = Module()
        m.submodules.dut = dut
        m.submodules.psram = _psram = FakePSRAM(
            addr_width=22,
            data_width=32,
            storage_words=storage_words,
            latency_cycles=4,
        )
        wiring.connect(m, dut.bus, _psram.bus)

        d = dut.dilation
        n = dut.num_entries
        results = []  # collect (step_i, output_payload) tuples

        async def testbench(ctx):
            ctx.set(dut.o.ready, 1)

            for step in range(n_steps):
                sample = np.array([0.01 * (j + 1) for j in range(in_out_d)])
                sample += step * 0.1
                sample_nnq = parse_nnq(list(sample), assert_exact=False)

                # Present input; keep valid until the DUT accepts it.
                ctx.set(dut.i.payload, sample_nnq)
                ctx.set(dut.i.valid, 1)

                accepted = False
                for _ in range(max_wait):
                    if ctx.get(dut.i.ready):
                        await ctx.tick()  # handshake tick
                        accepted = True
                        break
                    await ctx.tick()

                ctx.set(dut.i.valid, 0)
                self.assertTrue(accepted, f"Step {step}: input never accepted")

                # Wait for output.
                got_output = False
                for _ in range(max_wait):
                    if ctx.get(dut.o.valid):
                        got_output = True
                        break
                    await ctx.tick()

                self.assertTrue(got_output, f"Step {step}: no output produced")

                # Snapshot the output payload while o.valid is high.
                snapshot = []
                for k in range(K):
                    row = []
                    for dim in range(in_out_d):
                        row.append(ctx.get(dut.o.payload[k][dim]).as_float())
                    snapshot.append(row)
                results.append((step, snapshot))

                # Advance past the output cycle so o.valid de-asserts before
                # the next iteration sets i.valid again.
                await ctx.tick()

        sim = Simulator(m)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()
        return results

    def test_activation_cache_ps_basic(self):
        """
        Runs 20 steps with (in_out_d=3, dilation_level=1) and checks:
          - At every step, payload[3] == current input sample.
          - At step 19, the three historical taps match expected delayed values.

        Note: historical entries at step 0 are NOT checked because PSRAM
        memory is not zeroed on reset (unlike the EBR version).
        """
        results = self._run_activation_cache_ps(
            in_out_d=3,
            dilation_level=1,
            cache_kwargs={"burst_len": 4, "cachesize_words": 64},
            n_steps=20,
        )
        self.assertEqual(len(results), 20)

        for step, snapshot in results:
            sample = np.array([0.01, 0.02, 0.03]) + step * 0.1
            sample_nnq = parse_nnq(list(sample), assert_exact=False)

            # payload[3] must always equal the current (most recent) sample.
            for dim in range(3):
                self.assertAlmostEqual(
                    snapshot[3][dim],
                    sample_nnq[dim].as_float(),
                    msg=f"step={step} dim={dim}: payload[3] mismatch",
                )

        # At step 19 verify all four taps.
        # dilation=4, so the three read taps are 4, 8, 12 steps behind.
        step19 = dict(results)[19]

        def assert_tap(k, expecteds):
            nnq_exp = parse_nnq(expecteds, assert_exact=False)
            for dim in range(3):
                self.assertEqual(
                    step19[k][dim],
                    nnq_exp[dim].as_float(),
                    msg=f"step=19 k={k} dim={dim}: got {step19[k][dim]}, "
                    f"expected {nnq_exp[dim].as_float()}",
                )

        # 12 steps back from step 19 → step 7 → [0.71, 0.72, 0.73]
        assert_tap(k=0, expecteds=[0.71, 0.72, 0.73])
        # 8 steps back → step 11 → [1.11, 1.12, 1.13]
        assert_tap(k=1, expecteds=[1.11, 1.12, 1.13])
        # 4 steps back → step 15 → [1.51, 1.52, 1.53]
        assert_tap(k=2, expecteds=[1.51, 1.52, 1.53])
        # current → step 19 → [1.91, 1.92, 1.93]
        assert_tap(k=3, expecteds=[1.91, 1.92, 1.93])

    def test_activation_cache_ps_dilation2(self):
        """
        Smoke-test with dilation_level=2 (dilation=16, num_entries=64).
        Only checks that payload[3] always equals the current sample and
        that 20 outputs are produced without hanging.
        """
        results = self._run_activation_cache_ps(
            in_out_d=2,
            dilation_level=2,
            cache_kwargs={"burst_len": 4, "cachesize_words": 128},
            n_steps=20,
            max_wait=5000,
        )
        self.assertEqual(len(results), 20)

        for step, snapshot in results:
            sample = np.array([0.01, 0.02]) + step * 0.1
            sample_nnq = parse_nnq(list(sample), assert_exact=False)
            for dim in range(2):
                self.assertAlmostEqual(
                    snapshot[3][dim],
                    sample_nnq[dim].as_float(),
                    msg=f"step={step} dim={dim}: payload[3] mismatch",
                )
