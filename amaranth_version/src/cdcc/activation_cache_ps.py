"""
PSRAM-backed ActivationCache.

Replicates the behaviour of ActivationCache but uses external PSRAM via a
Wishbone bus and WishboneL2Cache (write-back, direct-mapped, burst cache).

Inlined components from tiliqua
  - WishboneL2Cache  (tiliqua/gateware/src/tiliqua/cache.py)
  - _WishboneAdapter (derived from tiliqua/gateware/src/tiliqua/dsp/delay_line.py)

Original copyright: (c) 2024 Seb Holzapfel <me@sebholzapfel.com>
Original SPDX-License-Identifier: CERN-OHL-S-2.0
"""

import math

from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out
from amaranth.utils import exact_log2
from amaranth_soc import wishbone

from . import NNQ, K


class WishboneL2Cache(wiring.Component):
    """
    Wishbone cache — direct-mapped, write-back.

    'master' bus: classic (non-burst) transactions from the client.
    'slave'  bus: burst transactions to the backing store (PSRAM).

    Inlined verbatim from tiliqua for self-contained use.
    """

    def __init__(
        self,
        cachesize_words=64,
        addr_width=22,
        data_width=32,
        granularity=8,
        burst_len=4,
        autoflush=False,
    ):
        assert burst_len > 1
        self.cachesize_words = cachesize_words
        self.data_width = data_width
        self.burst_len = burst_len
        self.granularity = granularity
        self.autoflush = autoflush
        super().__init__(
            {
                "master": In(
                    wishbone.Signature(
                        addr_width=addr_width,
                        data_width=data_width,
                        granularity=granularity,
                    )
                ),
                "slave": Out(
                    wishbone.Signature(
                        addr_width=addr_width,
                        data_width=data_width,
                        granularity=granularity,
                        features={"cti", "bte"},
                    )
                ),
            }
        )

    def elaborate(self, platform):
        m = Module()

        master = self.master
        slave = self.slave

        dw_from = dw_to = self.data_width

        addressbits = len(slave.adr)
        offsetbits = exact_log2(self.burst_len)
        linebits = exact_log2(self.cachesize_words // self.burst_len)
        tagbits = addressbits - linebits - offsetbits
        adr_offset = master.adr.bit_select(0, offsetbits)
        adr_line = Signal(linebits)
        adr_tag = master.adr.bit_select(offsetbits + linebits, tagbits)

        m.d.comb += adr_line.eq(master.adr.bit_select(offsetbits, linebits))

        burst_offset = Signal.like(adr_offset)
        burst_offset_lookahead = Signal.like(burst_offset)

        m.submodules.data_mem = data_mem = Memory(
            shape=unsigned(self.data_width), depth=2**linebits * self.burst_len, init=[]
        )
        wr_port = data_mem.write_port(granularity=self.granularity)
        rd_port = data_mem.read_port()

        write_from_slave = Signal()
        word_select = Const(1).replicate(dw_to // self.granularity)

        m.d.comb += [
            rd_port.addr.eq(Cat(adr_offset, adr_line)),
            slave.sel.eq(word_select),
            master.dat_r.eq(rd_port.data),
            slave.dat_w.eq(rd_port.data),
        ]

        with m.If(write_from_slave):
            m.d.comb += [
                wr_port.addr.eq(Cat(burst_offset, adr_line)),
                wr_port.data.eq(slave.dat_r),
                wr_port.en.eq(word_select),
            ]
        with m.Else():
            m.d.comb += wr_port.addr.eq(Cat(adr_offset, adr_line))
            m.d.comb += wr_port.data.eq(master.dat_w)
            with m.If(master.cyc & master.stb & master.we & master.ack):
                m.d.comb += wr_port.en.eq(master.sel)

        tag_layout = data.StructLayout(
            {
                "tag": unsigned(tagbits),
                "dirty": unsigned(1),
                "valid": unsigned(1),
            }
        )
        m.submodules.tag_mem = tag_mem = Memory(
            shape=tag_layout, depth=2**linebits, init=[]
        )
        tag_wr_port = tag_mem.write_port()
        tag_rd_port = tag_mem.read_port(domain="comb")
        tag_do = Signal(shape=tag_layout)
        tag_di = Signal(shape=tag_layout)
        m.d.comb += [
            tag_do.eq(tag_rd_port.data),
            tag_wr_port.data.eq(tag_di),
        ]
        m.d.comb += [
            tag_wr_port.addr.eq(adr_line),
            tag_rd_port.addr.eq(adr_line),
            tag_di.tag.eq(adr_tag),
        ]
        m.d.comb += slave.adr.eq(Cat(burst_offset, adr_line, tag_do.tag))
        m.d.sync += master.ack.eq(0)

        if self.autoflush:
            flush_wait = Signal(10, init=1)
            adr_line_flush = Signal.like(adr_line)

        with m.FSM() as fsm:

            with m.State("IDLE"):
                with m.If(master.cyc & master.stb):
                    m.next = "TEST_HIT"
                if self.autoflush:
                    m.d.sync += flush_wait.eq(flush_wait + 1)
                    with m.If(flush_wait == 0):
                        m.d.comb += adr_line.eq(adr_line_flush)
                        m.next = "TEST_FLUSH"

            with m.State("WAIT"):
                m.next = "IDLE"

            with m.State("TEST_HIT"):
                with m.If((tag_do.tag == adr_tag) & tag_do.valid):
                    m.d.sync += master.ack.eq(1)
                    with m.If(master.we):
                        m.d.comb += [
                            tag_di.valid.eq(1),
                            tag_di.dirty.eq(1),
                            tag_wr_port.en.eq(1),
                        ]
                    m.next = "WAIT"
                with m.Else():
                    with m.If(tag_do.dirty):
                        m.d.comb += rd_port.addr.eq(
                            Cat(burst_offset_lookahead, adr_line)
                        )
                        m.next = "EVICT"
                    with m.Else():
                        m.d.comb += [
                            tag_di.valid.eq(1),
                            tag_wr_port.en.eq(1),
                        ]
                        m.next = "REFILL"

            with m.State("EVICT"):
                m.d.comb += [
                    slave.stb.eq(1),
                    slave.cyc.eq(1),
                    slave.we.eq(1),
                    slave.cti.eq(wishbone.CycleType.INCR_BURST),
                    rd_port.addr.eq(Cat(burst_offset_lookahead, adr_line)),
                ]
                with m.If(slave.ack):
                    m.d.comb += burst_offset_lookahead.eq(burst_offset + 1)
                    m.d.sync += burst_offset.eq(burst_offset + 1)
                    with m.If(burst_offset == (self.burst_len - 1)):
                        m.d.comb += slave.cti.eq(wishbone.CycleType.END_OF_BURST)
                        m.next = "WAIT-REFILL"

            with m.State("WAIT-REFILL"):
                m.d.comb += [
                    tag_di.valid.eq(1),
                    tag_wr_port.en.eq(1),
                ]
                m.next = "REFILL"

            with m.State("REFILL"):
                m.d.comb += [
                    slave.stb.eq(1),
                    slave.cyc.eq(1),
                    slave.we.eq(0),
                    slave.cti.eq(wishbone.CycleType.INCR_BURST),
                ]
                with m.If(slave.ack):
                    m.d.comb += write_from_slave.eq(1)
                    m.d.sync += burst_offset.eq(burst_offset + 1)
                    with m.If(burst_offset == (self.burst_len - 1)):
                        m.d.comb += slave.cti.eq(wishbone.CycleType.END_OF_BURST)
                        m.next = "TEST_HIT"

            if self.autoflush:
                with m.State("TEST_FLUSH"):
                    m.d.comb += adr_line.eq(adr_line_flush)
                    with m.If(tag_do.valid & tag_do.dirty):
                        m.next = "FLUSH_LINE"
                    with m.Else():
                        m.d.sync += adr_line_flush.eq(adr_line_flush + 1)
                        m.next = "IDLE"

                with m.State("FLUSH_LINE"):
                    m.d.comb += [
                        adr_line.eq(adr_line_flush),
                        slave.stb.eq(1),
                        slave.cyc.eq(1),
                        slave.we.eq(1),
                        slave.cti.eq(wishbone.CycleType.INCR_BURST),
                        rd_port.addr.eq(Cat(burst_offset_lookahead, adr_line)),
                    ]
                    with m.If(slave.ack):
                        m.d.comb += burst_offset_lookahead.eq(burst_offset + 1)
                        m.d.sync += burst_offset.eq(burst_offset + 1)
                        with m.If(burst_offset == (self.burst_len - 1)):
                            m.d.comb += [
                                slave.cti.eq(wishbone.CycleType.END_OF_BURST),
                                tag_di.valid.eq(0),
                                tag_wr_port.en.eq(1),
                            ]
                            m.d.sync += adr_line_flush.eq(adr_line_flush + 1)
                            m.next = "IDLE"

        return m


class _WishboneAdapter(wiring.Component):
    """
    Adapter: 16-bit internal Wishbone <-> 32-bit external Wishbone.

    Two 16-bit samples share one 32-bit word (even address = low half,
    odd address = high half).  Base is a byte address into the external
    memory space; must be 4-byte aligned.
    """

    def __init__(self, addr_width_i, addr_width_o, base):
        self.base = base
        assert (base & 0x3) == 0, "base address must be 4-byte aligned"
        super().__init__(
            {
                "i": In(
                    wishbone.Signature(
                        addr_width=addr_width_i, data_width=16, granularity=8
                    )
                ),
                "o": Out(
                    wishbone.Signature(
                        addr_width=addr_width_o, data_width=32, granularity=8
                    )
                ),
            }
        )

    def elaborate(self, platform):
        m = Module()
        m.d.comb += [
            self.i.ack.eq(self.o.ack),
            self.o.adr.eq((self.base >> 2) + (self.i.adr >> 1)),
            self.o.we.eq(self.i.we),
            self.o.cyc.eq(self.i.cyc),
            self.o.stb.eq(self.i.stb),
        ]
        with m.If(self.i.adr[0]):
            m.d.comb += [
                self.i.dat_r.eq(self.o.dat_r >> 16),
                self.o.sel.eq(self.i.sel << 2),
                self.o.dat_w.eq(self.i.dat_w << 16),
            ]
        with m.Else():
            m.d.comb += [
                self.i.dat_r.eq(self.o.dat_r),
                self.o.sel.eq(self.i.sel),
                self.o.dat_w.eq(self.i.dat_w),
            ]
        return m


class ActivationCachePS(wiring.Component):
    """
    PSRAM-backed version of ActivationCache (write_triggers_read=True pattern).

    Stores a circular buffer of `num_entries = K * dilation` multi-dimensional
    NNQ samples in external PSRAM.  On each accepted input sample the component:

      1. Writes the new sample (all `in_out_d` dimensions) to PSRAM.
      2. Reads back the entries at delays d, 2d, 3d (one dimension per bus
         transaction, sequentially).
      3. Presents all four entries (delays 3d, 2d, d, 0) as output payload.

    Internal address layout (16-bit words, dimension-major):
      dimension k occupies addresses  [k * num_entries, (k+1) * num_entries)

    The internal 16-bit bus is bridged to the 32-bit external PSRAM bus via
    _WishboneAdapter (packs two 16-bit samples per 32-bit word) and then
    through WishboneL2Cache for burst efficiency.

    Members
    -------
    i   : In(stream.Signature(ArrayLayout(NNQ, in_out_d)))
    o   : Out(stream.Signature(ArrayLayout(ArrayLayout(NNQ, in_out_d), K)))
    bus : Out(wishbone.Signature(...))   – connect to PSRAM
    """

    def __init__(
        self,
        in_out_d: int,
        dilation_level: int,
        addr_width_o: int = 22,
        base: int = 0,
        cache_kwargs=None,
    ):
        if dilation_level < 1:
            raise ValueError(f"dilation_level must be >=1, received {dilation_level}")
        if K != 4:
            raise ValueError(f"ActivationCachePS is specialised for K=4, got K={K}")

        self.in_out_d = in_out_d
        self.input_layout = data.ArrayLayout(NNQ, in_out_d)
        self.output_layout = data.ArrayLayout(self.input_layout, K)

        self.dilation = K**dilation_level  # = K^level, always pow2
        self.num_entries = K * self.dilation  # = K^(level+1), always pow2
        self.address_width = exact_log2(self.num_entries)

        # Total 16-bit internal address space: one region of num_entries per dimension.
        # ceil(log2(d * 2^aw)) = ceil(log2(d)) + aw  (exact when d | 2^aw, conservative o/w)
        total_words = in_out_d * self.num_entries
        self.internal_addr_width = (
            int(math.ceil(math.log2(total_words))) if total_words > 1 else 1
        )

        # Pre-compute ring-buffer offsets for the three read taps (Python ints).
        # tap k reads at: (write_ptr + offset_k) & ring_mask
        # offset_k = num_entries - k*dilation  (positive modular equivalent of -k*d)
        self._off_k1 = self.num_entries - 1 * self.dilation
        self._off_k2 = self.num_entries - 2 * self.dilation
        self._off_k3 = self.num_entries - 3 * self.dilation

        # Internal 16-bit Wishbone bus (master, driven by the FSM below).
        self._bus = wishbone.Signature(
            addr_width=self.internal_addr_width,
            data_width=16,
            granularity=8,
        ).create()

        # 16-bit → 32-bit adapter.
        self._adapter = _WishboneAdapter(
            addr_width_i=self.internal_addr_width,
            addr_width_o=addr_width_o,
            base=base,
        )

        # L2 write-back cache between adapter and PSRAM.
        _ck = cache_kwargs if cache_kwargs is not None else {}
        self._cache = WishboneL2Cache(addr_width=addr_width_o, **_ck)

        print(
            f">ActivationCachePS in_out_d={in_out_d}"
            f" dilation_level={dilation_level}"
            f" => dilation={self.dilation}"
            f" => |entries|={self.num_entries}"
            f" => internal_addr_width={self.internal_addr_width}"
        )

        super().__init__(
            {
                "i": In(stream.Signature(self.input_layout)),
                "o": Out(stream.Signature(self.output_layout)),
                "bus": Out(
                    wishbone.Signature(
                        addr_width=addr_width_o,
                        data_width=32,
                        granularity=8,
                        features={"bte", "cti"},
                    )
                ),
            }
        )

    def elaborate(self, platform):
        m = Module()

        m.submodules.adapter = self._adapter
        m.submodules.cache = self._cache

        # Connect: _bus (master FSM) --manual--> adapter.i --wiring--> cache.master
        #          cache.slave --wiring--> self.bus (external PSRAM port)
        bus = self._bus
        m.d.comb += [
            self._adapter.i.stb.eq(bus.stb),
            self._adapter.i.cyc.eq(bus.cyc),
            self._adapter.i.we.eq(bus.we),
            self._adapter.i.adr.eq(bus.adr),
            self._adapter.i.dat_w.eq(bus.dat_w),
            self._adapter.i.sel.eq(bus.sel),
            bus.dat_r.eq(self._adapter.i.dat_r),
            bus.ack.eq(self._adapter.i.ack),
        ]
        wiring.connect(m, self._adapter.o, self._cache.master)
        wiring.connect(m, self._cache.slave, wiring.flipped(self.bus))

        # ---- Datapath signals -----------------------------------------------
        NNQ_BITS = NNQ.width  # 16
        n = self.num_entries
        ring_mask = n - 1
        aw = self.address_width

        write_ptr = Signal(range(n), init=0)
        accepted_ptr = Signal(range(n), init=0)  # write_ptr at input acceptance

        # dim_counter iterates over 0..in_out_d-1 during sequential bus accesses.
        dim_counter = Signal(range(self.in_out_d))

        input_latch = Signal(self.input_layout)

        # Buffers for the three read taps (indexed by delay level: 0=d, 1=2d, 2=3d).
        tap_buf = [Signal(self.input_layout, name=f"tap_buf_{i}") for i in range(3)]

        # ---- FSM ------------------------------------------------------------
        #
        # WAIT-VALID  : accept input, latch payload and write_ptr
        # WRITE       : write input_latch[dim] to PSRAM for each dim in turn
        # READ-K3     : read (accepted_ptr - 3d) for each dim -> tap_buf[2]
        # READ-K2     : read (accepted_ptr - 2d) for each dim -> tap_buf[1]
        # READ-K1     : read (accepted_ptr - d)  for each dim -> tap_buf[0]
        # OUTPUT      : assert o.valid, wait for o.ready

        with m.FSM(domain="sync"):

            with m.State("WAIT-VALID"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.sync += [
                        input_latch.eq(self.i.payload),
                        accepted_ptr.eq(write_ptr),
                        dim_counter.eq(0),
                    ]
                    m.next = "WRITE"

            with m.State("WRITE"):
                m.d.comb += [
                    bus.stb.eq(1),
                    bus.cyc.eq(1),
                    bus.we.eq(1),
                    bus.sel.eq(-1),
                    bus.adr.eq((dim_counter << aw) | write_ptr),
                    bus.dat_w.eq(
                        input_latch.as_value().word_select(dim_counter, NNQ_BITS)
                    ),
                ]
                with m.If(bus.ack):
                    with m.If(dim_counter == self.in_out_d - 1):
                        m.d.sync += [
                            write_ptr.eq((write_ptr + 1) & ring_mask),
                            dim_counter.eq(0),
                        ]
                        m.next = "READ-K3"
                    with m.Else():
                        m.d.sync += dim_counter.eq(dim_counter + 1)

            with m.State("READ-K3"):
                m.d.comb += [
                    bus.stb.eq(1),
                    bus.cyc.eq(1),
                    bus.we.eq(0),
                    bus.sel.eq(-1),
                    bus.adr.eq(
                        (dim_counter << aw)
                        | ((accepted_ptr + self._off_k3) & ring_mask)
                    ),
                ]
                with m.If(bus.ack):
                    m.d.sync += (
                        tap_buf[2]
                        .as_value()
                        .word_select(dim_counter, NNQ_BITS)
                        .eq(bus.dat_r)
                    )
                    with m.If(dim_counter == self.in_out_d - 1):
                        m.d.sync += dim_counter.eq(0)
                        m.next = "READ-K2"
                    with m.Else():
                        m.d.sync += dim_counter.eq(dim_counter + 1)

            with m.State("READ-K2"):
                m.d.comb += [
                    bus.stb.eq(1),
                    bus.cyc.eq(1),
                    bus.we.eq(0),
                    bus.sel.eq(-1),
                    bus.adr.eq(
                        (dim_counter << aw)
                        | ((accepted_ptr + self._off_k2) & ring_mask)
                    ),
                ]
                with m.If(bus.ack):
                    m.d.sync += (
                        tap_buf[1]
                        .as_value()
                        .word_select(dim_counter, NNQ_BITS)
                        .eq(bus.dat_r)
                    )
                    with m.If(dim_counter == self.in_out_d - 1):
                        m.d.sync += dim_counter.eq(0)
                        m.next = "READ-K1"
                    with m.Else():
                        m.d.sync += dim_counter.eq(dim_counter + 1)

            with m.State("READ-K1"):
                m.d.comb += [
                    bus.stb.eq(1),
                    bus.cyc.eq(1),
                    bus.we.eq(0),
                    bus.sel.eq(-1),
                    bus.adr.eq(
                        (dim_counter << aw)
                        | ((accepted_ptr + self._off_k1) & ring_mask)
                    ),
                ]
                with m.If(bus.ack):
                    m.d.sync += (
                        tap_buf[0]
                        .as_value()
                        .word_select(dim_counter, NNQ_BITS)
                        .eq(bus.dat_r)
                    )
                    with m.If(dim_counter == self.in_out_d - 1):
                        m.next = "OUTPUT"
                    with m.Else():
                        m.d.sync += dim_counter.eq(dim_counter + 1)

            with m.State("OUTPUT"):
                # payload[0] = oldest (3d behind), payload[3] = current — matches
                # the ordering produced by the EBR ActivationCache.
                m.d.comb += [
                    self.o.valid.eq(1),
                    self.o.payload[0].eq(tap_buf[2]),  # delay 3d
                    self.o.payload[1].eq(tap_buf[1]),  # delay 2d
                    self.o.payload[2].eq(tap_buf[0]),  # delay  d
                    self.o.payload[3].eq(input_latch),  # current
                ]
                with m.If(self.o.ready):
                    m.d.sync += dim_counter.eq(0)
                    m.next = "WAIT-VALID"

        return m
