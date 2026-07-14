import math

from amaranth.lib import stream, wiring, data
from amaranth.lib.memory import Memory
from amaranth import Module, Signal, unsigned

from . import NNQ, parse_nnq


class QuadratureGenerator(wiring.Component):

    # given a specified sample_rate and fixed freq_hz, setup a memoryfor
    # sine wave ( of amplitude 0.8 ) and use it to emit (sin, cos) pair each clock

    def __init__(self, sample_rate: int, freq_hz: float, amplitude: float):
        self.sample_rate = sample_rate
        self.freq_hz = freq_hz
        self.amplitude = amplitude
        self.lut_size = max(1, round(sample_rate / freq_hz))
        just_handshake_input = unsigned(0)
        super().__init__(
            {
                "i": wiring.In(stream.Signature(just_handshake_input)),
                "o": wiring.Out(stream.Signature(data.ArrayLayout(NNQ, 2))),
            }
        )

    def elaborate(self, platform):
        m = Module()

        sin_init = [
            parse_nnq(
                self.amplitude * math.sin(2 * math.pi * n / self.lut_size),
                assert_exact=False,
                shape=NNQ,
            )
            for n in range(self.lut_size)
        ]
        m.submodules.sin_mem = sin_mem = Memory(
            shape=NNQ,
            depth=self.lut_size,
            init=sin_init,
            attrs={"ram_style": "block"},
        )
        rd_sin = sin_mem.read_port(domain="sync")
        rd_cos = sin_mem.read_port(domain="sync")

        cos_offset = round(self.lut_size / 4)
        idx = Signal(range(self.lut_size), init=0)

        cos_idx = Signal(range(self.lut_size))
        with m.If(idx + cos_offset >= self.lut_size):
            m.d.comb += cos_idx.eq(idx + cos_offset - self.lut_size)
        with m.Else():
            m.d.comb += cos_idx.eq(idx + cos_offset)

        # just constantly read and set out payload
        m.d.comb += [
            rd_sin.addr.eq(idx),
            rd_cos.addr.eq(cos_idx),
            self.o.payload[0].eq(rd_sin.data),
            self.o.payload[1].eq(rd_cos.data),
        ]

        with m.FSM():

            # TODO: why is this required? just a _d effect?
            with m.State("WAIT"):
                m.next = "VALID"

            with m.State("VALID"):
                m.d.comb += [
                    self.o.valid.eq(self.i.valid),
                    self.i.ready.eq(self.o.ready),
                ]
                with m.If(self.i.valid & self.o.ready):
                    with m.If(idx == self.lut_size - 1):
                        m.d.sync += idx.eq(0)
                    with m.Else():
                        m.d.sync += idx.eq(idx + 1)
                    m.next = "WAIT"

        return m
