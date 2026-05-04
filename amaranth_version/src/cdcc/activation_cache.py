from amaranth import Array, Module, Mux, Signal
from amaranth.lib import data, stream, wiring

from . import NNQ, K


class ActivationCache(wiring.Component):
    # note: for dilation=1 this is equivalent to left_shift_buffer

    def __init__(self, in_out_d: int, dilation_level: int):
        if dilation_level < 1:
            raise ValueError(f"dilation_level must be >=1, received {dilation_level}")

        self.in_out_d = in_out_d

        super().__init__(
            {
                "i": wiring.In(stream.Signature(data.ArrayLayout(NNQ, self.in_out_d))),
                "o": wiring.Out(
                    stream.Signature(
                        data.ArrayLayout(data.ArrayLayout(NNQ, self.in_out_d), K)
                    )
                ),
            }
        )

        self._dilation = K**dilation_level
        self._num_entries = K * self._dilation

        feature = data.ArrayLayout(NNQ, self.in_out_d)
        self._buffer = Array(
            Signal(feature, name=f"ac_{idx}", init=[0] * self.in_out_d)
            for idx in range(self._num_entries)
        )
        self._write_head = Signal(range(self._num_entries), init=0)

    def elaborate(self, platform):
        assert K == 4, "code assumes K=4"

        m = Module()

        d = self._dilation
        n = self._num_entries

        idx_1d = Signal(range(n))
        idx_2d = Signal(range(n))
        idx_3d = Signal(range(n))

        m.d.comb += [
            self.i.ready.eq(self.o.ready),
            self.o.valid.eq(self.i.valid),
            idx_1d.eq(
                Mux(
                    self._write_head >= d,
                    self._write_head - d,
                    self._write_head + (n - d),
                )
            ),
            idx_2d.eq(
                Mux(
                    self._write_head >= 2 * d,
                    self._write_head - (2 * d),
                    self._write_head + (n - (2 * d)),
                )
            ),
            idx_3d.eq(
                Mux(
                    self._write_head >= 3 * d,
                    self._write_head - (3 * d),
                    self._write_head + (n - (3 * d)),
                )
            ),
            self.o.payload[0].eq(self._buffer[idx_3d]),
            self.o.payload[1].eq(self._buffer[idx_2d]),
            self.o.payload[2].eq(self._buffer[idx_1d]),
            self.o.payload[3].eq(self.i.payload),
        ]

        with m.If(self.i.valid & self.i.ready):
            m.d.sync += [
                self._buffer[self._write_head].eq(self.i.payload),
                self._write_head.eq(
                    Mux(self._write_head == (n - 1), 0, self._write_head + 1)
                ),
            ]

        return m
