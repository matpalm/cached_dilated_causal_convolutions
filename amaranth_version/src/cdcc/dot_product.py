import json

from amaranth import Array, Module, Signal
from amaranth.lib import data, stream, wiring

from amaranth_future import fixed

# from . import N_FRAC, N_INT, NNQ, NNQ_DW
from . import NNQ, NNQ_DW, parse_nnq

class DotProduct(wiring.Component):
    """Compute dot product between input vector and fixed weights.

    The implementation mirrors the staged multiply-accumulate behavior used
    in the SystemVerilog module.
    """

    # OUT_Q = fixed.SQ(2 * N_INT, 2 * N_FRAC)

    def __init__(self, weights):
        self._weights = self._parse_weights(weights)
        self.D = len(self._weights)

        if self.D == 0:
            raise ValueError("weights must contain at least one element")

        super().__init__(
            {
                "i": wiring.In(stream.Signature(data.ArrayLayout(NNQ, self.D))),
                "o": wiring.Out(stream.Signature(NNQ_DW)),
            }
        )

        self._state = Signal(2, init=0)
        self._index = Signal(range(self.D + 1), init=0)

        self._accumulator = Signal(NNQ_DW, init=0)
        self._product = Signal(NNQ_DW, init=0)
        self._a_values = Array(
            Signal(NNQ, name=f"a_{i}", init=0) for i in range(self.D)
        )

    def _parse_weights(self, weights):
        if isinstance(weights, str):
            with open(weights, "r") as w:
                str_weights = json.load(w)
            weights = map(float, str_weights)
        fp_weights = parse_nnq(weights, check_exact=True)
        return Array(fp_weights)

    def elaborate(self, platform):
        m = Module()

        STATE_IDLE = 0
        STATE_MULTIPLY_ELEMENT = 1
        STATE_DONE = 2

        m.d.comb += [
            self.i.ready.eq(self._state == STATE_IDLE),
            self.o.payload.eq(self._accumulator + self._product),
            self.o.valid.eq(self._state == STATE_DONE),
        ]

        with m.If(self._state == STATE_IDLE):
            with m.If(self.i.valid & self.i.ready):
                for j in range(self.D):
                    m.d.sync += self._a_values[j].eq(self.i.payload[j])
                m.d.sync += [
                    self._accumulator.eq(0),
                    self._product.eq(0),
                    self._index.eq(0),
                    self._state.eq(STATE_MULTIPLY_ELEMENT),
                ]

        with m.Elif(self._state == STATE_MULTIPLY_ELEMENT):

            m.d.sync += [
                self._accumulator.eq(self._accumulator + self._product),
                self._product.eq(
                    (
                        NNQ(self._a_values[self._index])
                        * NNQ(self._weights[self._index])
                    ).as_value()
                ),
                self._index.eq(self._index + 1),
            ]

            with m.If(self._index == self.D - 1):
                m.d.sync += self._state.eq(STATE_DONE)

        with m.Elif(self._state == STATE_DONE):
            with m.If(self.o.ready):
                m.d.sync += [
                    self._state.eq(STATE_IDLE),
                ]

        return m
