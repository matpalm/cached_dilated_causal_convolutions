from amaranth import Array, Elaboratable, Module, Signal, signed


class LeftShiftBuffer(Elaboratable):
    """4-tap shift buffer matching `src/left_shift_buffer.sv` behavior."""

    def __init__(self, w=16):
        if w <= 0:
            raise ValueError("w must be > 0")

        self.w = w

        self.rst = Signal()
        self.inp = Signal(signed(w))

        self.out_0 = Signal(signed(w))
        self.out_1 = Signal(signed(w))
        self.out_2 = Signal(signed(w))
        self.out_3 = Signal(signed(w))

        self._buffer = Array(
            Signal(signed(w), name=f"buffer_{idx}", init=0) for idx in range(4)
        )

    def elaborate(self, platform):
        m = Module()

        m.d.comb += [
            self.out_0.eq(self._buffer[0]),
            self.out_1.eq(self._buffer[1]),
            self.out_2.eq(self._buffer[2]),
            self.out_3.eq(self._buffer[3]),
        ]

        with m.If(self.rst):
            for i in range(4):
                m.d.sync += self._buffer[i].eq(0)
        with m.Else():
            m.d.sync += [
                self._buffer[0].eq(self._buffer[1]),
                self._buffer[1].eq(self._buffer[2]),
                self._buffer[2].eq(self._buffer[3]),
                self._buffer[3].eq(self.inp),
            ]

        return m
