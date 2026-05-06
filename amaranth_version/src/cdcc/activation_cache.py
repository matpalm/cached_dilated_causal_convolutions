from amaranth import Array, Cat, Module, Signal
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import Memory

from . import NNQ, K


class ActivationCache(wiring.Component):
    # note: for dilation=1 this is equivalent to left_shift_buffer

    def __init__(self, in_out_d: int, dilation_level: int, use_ebr: bool = False):
        if dilation_level < 1:
            raise ValueError(f"dilation_level must be >=1, received {dilation_level}")
        if K != 4:
            raise ValueError(f"ActivationCache is specialized for K=4, received {K}")

        self.in_out_d = in_out_d
        self.input_layout = data.ArrayLayout(NNQ, self.in_out_d)
        self.output_layout = data.ArrayLayout(self.input_layout, K)

        super().__init__(
            {
                "i": wiring.In(stream.Signature(self.input_layout)),
                "o": wiring.Out(stream.Signature(self.output_layout)),
            }
        )

        self.dilation = K**dilation_level
        self.num_entries = K * self.dilation
        self.use_ebr = use_ebr

        feature = self.input_layout
        self.buffer = None
        self.ebr_memories = None

        if self.use_ebr:
            init = [[0] * self.in_out_d for _ in range(self.num_entries)]
            self.ebr_memories = [
                Memory(
                    shape=feature,
                    depth=self.num_entries,
                    init=init,
                    attrs={"ram_style": "block"},
                )
                for i in range(3)
            ]
        else:
            self.buffer = Array(
                Signal(feature, name=f"ac_{idx}", init=[0] * self.in_out_d)
                for idx in range(self.num_entries)
            )
        self.write_head = Signal(range(self.num_entries), init=0)

    def elaborate(self, platform):
        m = Module()

        d = self.dilation
        n = self.num_entries
        ring_mask = n - 1

        idx_1d = Signal(range(n))
        idx_2d = Signal(range(n))
        idx_3d = Signal(range(n))

        m.d.comb += [
            self.i.ready.eq(self.o.ready),
            self.o.valid.eq(self.i.valid),
            # n is always a power-of-two (K=4 and dilation is K**level),
            # so modulo-n wrap is just masking the low bits.
            idx_1d.eq((self.write_head - d) & ring_mask),
            idx_2d.eq((self.write_head - (2 * d)) & ring_mask),
            idx_3d.eq((self.write_head - (3 * d)) & ring_mask),
            self.o.payload[3].eq(self.i.payload),
        ]

        if self.use_ebr:
            for i, mem in enumerate(self.ebr_memories):
                m.submodules[f"ac_mem_{i}"] = mem

            rd_1d = self.ebr_memories[0].read_port(domain="comb")
            rd_2d = self.ebr_memories[1].read_port(domain="comb")
            rd_3d = self.ebr_memories[2].read_port(domain="comb")

            wr_ports = [mem.write_port(domain="sync") for mem in self.ebr_memories]

            m.d.comb += [
                rd_1d.addr.eq(idx_1d),
                rd_2d.addr.eq(idx_2d),
                rd_3d.addr.eq(idx_3d),
                self.o.payload[0].eq(rd_3d.data),
                self.o.payload[1].eq(rd_2d.data),
                self.o.payload[2].eq(rd_1d.data),
            ]

            for wr in wr_ports:
                m.d.comb += [
                    wr.addr.eq(self.write_head),
                    wr.data.eq(self.i.payload),
                    wr.en.eq(self.i.valid & self.i.ready),
                ]
        else:
            m.d.comb += [
                self.o.payload[0].eq(self.buffer[idx_3d]),
                self.o.payload[1].eq(self.buffer[idx_2d]),
                self.o.payload[2].eq(self.buffer[idx_1d]),
            ]

            with m.If(self.i.valid & self.i.ready):
                m.d.sync += self.buffer[self.write_head].eq(self.i.payload)

        with m.If(self.i.valid & self.i.ready):
            m.d.sync += self.write_head.eq((self.write_head + 1) & ring_mask)

        return m
