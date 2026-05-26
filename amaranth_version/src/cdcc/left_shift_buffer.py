from amaranth import Array, Elaboratable, Module, Signal
from amaranth.lib import data, stream, wiring

from . import NNQ, K

class LeftShiftBuffer(wiring.Component):
    """Shift register like buffer For input.

    Prepares left shifting input feature into a length K=4 time series
    as input to first 1D convolution.
    """

    def __init__(self, in_out_d: int):

        feature = data.ArrayLayout(NNQ, in_out_d)
        self.buffer = Array(
            Signal(feature, name=f"lsb_{k}", init=[0] * in_out_d) for k in range(K)
        )

        super().__init__(
            {
                "i": wiring.In(stream.Signature(data.ArrayLayout(NNQ, in_out_d))),
                "o": wiring.Out(
                    stream.Signature(
                        data.ArrayLayout(data.ArrayLayout(NNQ, in_out_d), K)
                    )
                ),
            }
        )

    def elaborate(self, platform):
        m = Module()

        m.d.comb += [
            self.i.ready.eq(self.o.ready),
            self.o.valid.eq(self.i.valid),
        ]

        with m.If(self.i.valid & self.i.ready):
            for k in range(K - 1):
                m.d.sync += self.buffer[k].eq(self.buffer[k + 1])
            m.d.sync += self.buffer[K - 1].eq(self.i.payload)

        for k in range(K - 1):
            m.d.comb += self.o.payload[k].eq(self.buffer[k + 1])
        m.d.comb += self.o.payload[K - 1].eq(self.i.payload)

        return m
