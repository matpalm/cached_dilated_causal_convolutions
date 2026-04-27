from amaranth import Array, Elaboratable, Module, Signal
from amaranth.lib import data, stream, wiring

from . import NNQ


class LeftShiftBuffer(wiring.Component):
    """Shift register like buffer For input.

    Prepares left shifting input feature into a length K=4 time series
    as input to first 1D convolution.
    """

    IN_D = 4  # input dim; (x, e0, e1, 0)
    OUT_D = 4  # output dim; (x, e0, e1, 0)
    K = 4  # kernel size

    i: wiring.In(stream.Signature(data.ArrayLayout(NNQ, IN_D)))
    o: wiring.Out(stream.Signature(data.ArrayLayout(data.ArrayLayout(NNQ, OUT_D), K)))

    def __init__(self):
        super().__init__()

        feature = data.ArrayLayout(NNQ, self.OUT_D)
        self.buffer = Array(
            Signal(feature, name=f"lsb_{k}", init=[0] * self.OUT_D)
            for k in range(self.K)
        )

    def elaborate(self, platform):
        m = Module()

        m.d.comb += [
            self.i.ready.eq(self.o.ready),
            self.o.valid.eq(self.i.valid),
        ]

        with m.If(self.i.valid & self.i.ready):
            for k in range(self.K - 1):
                m.d.sync += self.buffer[k].eq(self.buffer[k + 1])
            m.d.sync += self.buffer[self.K - 1].eq(self.i.payload)

        for k in range(self.K):
            m.d.comb += self.o.payload[k].eq(self.buffer[k])

        return m
