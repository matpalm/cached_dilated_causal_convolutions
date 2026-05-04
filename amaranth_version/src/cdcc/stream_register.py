from amaranth import Module, Signal
from amaranth.lib import stream, wiring


class StreamRegister(wiring.Component):
    """Single-entry register stage for ready/valid streams."""

    def __init__(self, payload_shape):
        super().__init__(
            {
                "i": wiring.In(stream.Signature(payload_shape)),
                "o": wiring.Out(stream.Signature(payload_shape)),
            }
        )

        self._payload = Signal(payload_shape)
        self._valid = Signal(init=0)

    def elaborate(self, platform):
        m = Module()

        m.d.comb += [
            self.i.ready.eq(~self._valid | self.o.ready),
            self.o.valid.eq(self._valid),
            self.o.payload.eq(self._payload),
        ]

        with m.If(self.i.valid & self.i.ready):
            m.d.sync += [
                self._payload.eq(self.i.payload),
                self._valid.eq(1),
            ]
        with m.Elif(self.o.ready):
            m.d.sync += self._valid.eq(0)

        return m
