from amaranth import (
    Array,
    Const,
    Elaboratable,
    Module,
    Mux,
    Signal,
)


class ActivationCache(Elaboratable):
    """Activation cache for a 1D dilated convolution kernel.

    This mirrors the behavior of `src/activation_cache.sv` in `sverilog_version`:
    - Circular buffer depth is `dilation * kernel_size`.
    - On each clock:
      - write `inp` at `write_head`
      - increment `write_head` (with wrap)
      - update outputs from taps at 3*dilation, 2*dilation, 1*dilation steps back
      - `out_l3` is passthrough of current `inp`
    """

    def __init__(self, w=16, d=2, dilation=4, kernel_size=4):
        if w <= 0:
            raise ValueError("w must be > 0")
        if d <= 0:
            raise ValueError("d must be > 0")
        if dilation <= 0:
            raise ValueError("dilation must be > 0")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be > 0")

        self.w = w
        self.d = d
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.packed_width = w * d
        self.num_entries = dilation * kernel_size

        self.inp = Signal(self.packed_width)
        self.out_l0 = Signal(self.packed_width)
        self.out_l1 = Signal(self.packed_width)
        self.out_l2 = Signal(self.packed_width)
        self.out_l3 = Signal(self.packed_width)

        self._write_head = Signal(range(self.num_entries), init=0)
        self._buffer = Array(
            Signal(self.packed_width, name=f"buffer_{idx}", init=0)
            for idx in range(self.num_entries)
        )

    def elaborate(self, platform):
        m = Module()

        n = self.num_entries
        d = self.dilation

        out1_addr = Signal(range(n))
        out2_addr = Signal(range(n))
        out3_addr = Signal(range(n))

        m.d.comb += [
            out1_addr.eq((self._write_head + Const(n - (d % n), range(n))) % n),
            out2_addr.eq((self._write_head + Const(n - ((2 * d) % n), range(n))) % n),
            out3_addr.eq((self._write_head + Const(n - ((3 * d) % n), range(n))) % n),
        ]

        m.d.sync += [
            self._write_head.eq(
                Mux(self._write_head == (n - 1), 0, self._write_head + 1)
            ),
            self._buffer[self._write_head].eq(self.inp),
            self.out_l0.eq(self._buffer[out3_addr]),
            self.out_l1.eq(self._buffer[out2_addr]),
            self.out_l2.eq(self._buffer[out1_addr]),
            self.out_l3.eq(self.inp),
        ]

        return m
