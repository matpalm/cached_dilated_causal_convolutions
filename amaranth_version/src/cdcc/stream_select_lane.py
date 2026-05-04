from amaranth import Module
from amaranth.lib import data, stream, wiring

from . import NNQ


class StreamSelectLane(wiring.Component):
    def __init__(self, in_d: int, lane_index: int):
        if in_d < 1:
            raise ValueError(f"in_d must be >= 1, received {in_d}")
        if lane_index < 0 or lane_index >= in_d:
            raise ValueError(
                f"lane_index must be in [0, {in_d - 1}], received {lane_index}"
            )

        self.in_d = in_d
        self.lane_index = lane_index

        super().__init__(
            {
                "i": wiring.In(stream.Signature(data.ArrayLayout(NNQ, self.in_d))),
                "o": wiring.Out(stream.Signature(NNQ)),
            }
        )

    def elaborate(self, platform):
        m = Module()

        m.d.comb += [
            self.i.ready.eq(self.o.ready),
            self.o.valid.eq(self.i.valid),
            self.o.payload.eq(self.i.payload[self.lane_index]),
        ]

        return m
