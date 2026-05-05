import json

from amaranth import Array, Module, Signal
from amaranth.lib import data, stream, wiring
from amaranth_future import fixed
import numpy as np
from numpy.typing import NDArray

from . import NNQ, NNQ_DW, parse_nnq


class DotProduct(wiring.Component):

    def __init__(self, np_weights: NDArray):

        if len(np_weights.shape) != 1:
            raise Exception(
                f"Expect DotProduct to be inited with (D,) vector but received shape {np_weights.shape}"
            )

        self.D = np_weights.shape[0]

        super().__init__(
            {
                "i": wiring.In(stream.Signature(data.ArrayLayout(NNQ, self.D))),
                "o": wiring.Out(stream.Signature(NNQ_DW)),
            }
        )

        self._weights = Array(parse_nnq(np_weights))
        self._index = Signal(range(self.D + 1), init=0)

        # running dot-product sum is double width
        self._accumulator = Signal(NNQ_DW, init=0)

        self._a_values = Array(
            Signal(NNQ, name=f"a_{i}", init=0) for i in range(self.D)
        )

    def elaborate(self, platform):
        m = Module()

        m.d.comb += [
            self.i.ready.eq(0),
            self.o.valid.eq(0),
            self.o.payload.eq(self._accumulator),
        ]

        with m.FSM() as fsm:

            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid & self.i.ready):
                    for j in range(self.D):
                        m.d.sync += self._a_values[j].eq(self.i.payload[j])
                    m.d.sync += [
                        self._accumulator.eq(0),
                        self._index.eq(0),
                    ]
                    m.next = "MULTIPLY_ELEMENT"

            with m.State("MULTIPLY_ELEMENT"):
                # note: originally had accum in seperate assignment from product but combining
                # them allows yosys to know it's a MAC op and can use the accum in MULT18x18D
                # ( which results in TRELLIS_FF & TRELLIS_COMB dropping )
                m.d.sync += [
                    self._accumulator.eq(
                        self._accumulator.as_value().as_signed()
                        + (
                            self._a_values[self._index].as_value().as_signed()
                            * self._weights[self._index].as_value().as_signed()
                        )
                    ),
                    self._index.eq(self._index + 1),
                ]
                with m.If(self._index == self.D - 1):
                    m.next = "DONE"

            with m.State("DONE"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
