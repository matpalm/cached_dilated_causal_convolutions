from amaranth import Module, Signal
from amaranth.lib import stream, wiring


class StreamCut(wiring.Component):

    def __init__(self, payload_shape):
        super().__init__(
            {
                "i": wiring.In(stream.Signature(payload_shape)),
                "o": wiring.Out(stream.Signature(payload_shape)),
            }
        )

        self.payload = Signal(payload_shape)
        self.valid = Signal(init=0)

    def elaborate(self, platform):
        m = Module()

        m.d.comb += [
            self.i.ready.eq(~self.valid | self.o.ready),
            self.o.valid.eq(self.valid),
            self.o.payload.eq(self.payload),
        ]

        with m.If(self.i.valid & self.i.ready):
            m.d.sync += [
                self.payload.eq(self.i.payload),
                self.valid.eq(1),
            ]
        with m.Elif(self.o.ready):
            m.d.sync += self.valid.eq(0)

        return m
