from amaranth import Array, Cat, Module, Mux, Signal
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import Memory

from . import NNQ, K


class ActivationCache(wiring.Component):
    # note: for dilation=1 this is equivalent to left_shift_buffer

    # TODO: remove the use_ebr=False paths; we _always_ use it now

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
        print(
            f">ActivationCache use_ebr={use_ebr} in_out_d={in_out_d}"
            f" dilation_level={dilation_level} => dilation={self.dilation} => |entries|={self.num_entries}"
        )

        feature = self.input_layout
        self.buffer = None
        self.ebr_memory = None

        if self.use_ebr:
            init = [[0] * self.in_out_d for _ in range(self.num_entries)]
            self.ebr_memory = Memory(
                shape=feature,
                depth=self.num_entries,
                init=init,
                attrs={"ram_style": "block"},
            )
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
        ring_mask = n - 1  # assumes dilation always pow2 ( which is safe? )

        idx = Array(Signal(range(n), name=f"idx_{i}") for i in range(3))

        m.d.comb += [
            # n is always a power-of-two (K=4 and dilation is K**level),
            # so modulo-n wrap is just masking the low bits.
            idx[0].eq((self.write_head - d) & ring_mask),
            idx[1].eq((self.write_head - (2 * d)) & ring_mask),
            idx[2].eq((self.write_head - (3 * d)) & ring_mask),
        ]

        if self.use_ebr:
            m.submodules["ac_mem"] = self.ebr_memory

            # read and write
            rd = self.ebr_memory.read_port(domain="sync")
            wr = self.ebr_memory.write_port(domain="sync")

            # incoming value and write head
            accepted_payload = Signal(self.input_layout)
            accepted_head = Signal(range(n), init=0)

            # three outputs taps and idxs for reads
            tap = Array(Signal(self.input_layout, name=f"tap_{i}") for i in range(3))
            idx_accepted = Array(
                Signal(range(n), name=f"idx_{i}_accepted") for i in range(3)
            )

            m.d.comb += [
                idx_accepted[0].eq((accepted_head - d) & ring_mask),
                idx_accepted[1].eq((accepted_head - (2 * d)) & ring_mask),
                idx_accepted[2].eq((accepted_head - (3 * d)) & ring_mask),
                self.o.payload[0].eq(tap[2]),
                self.o.payload[1].eq(tap[1]),
                self.o.payload[2].eq(tap[0]),
                self.o.payload[3].eq(accepted_payload),
                wr.addr.eq(self.write_head),
                wr.data.eq(self.i.payload),
                wr.en.eq(self.i.valid & self.i.ready),
                rd.addr.eq(0),
                rd.en.eq(0),
                self.i.ready.eq(0),
                self.o.valid.eq(0),
            ]

            # cycle	state		i.r	i.v	o.v	o.r
            # 0	    IDLE		1	1	0	1	accept input, capture payload -> READ3
            # 1	    READ3		0	X	0	1	set read for tap3 -> READ2
            # 2     READ2		0	X	0	1	capture tap3, set read for tap2 -> READ1
            # 3	    READ1		0	X	0	1	capture tap2, set read for tap1 -> OUTPREP
            # 4	    OUT_PREP	0	X	0	1	capture tap1 -> OUTPUT
            # 5	    OUTPUT	    0	X	1	1	output valid, wait for o.ready -> IDLE

            with m.FSM(domain="sync", reset="IDLE") as fsm:

                with m.State("IDLE"):
                    # ready to process next input
                    with m.If(self.i.valid & self.i.ready):
                        m.d.sync += [
                            accepted_payload.eq(self.i.payload),
                            accepted_head.eq(self.write_head),
                        ]
                        m.next = "READ3"
                    m.d.comb += [
                        self.i.ready.eq(1),
                    ]

                with m.State("READ3"):
                    # prep for read 2
                    # ( note read3 done explicitly with incoming payload )
                    m.d.comb += [rd.en.eq(1)]
                    m.next = "READ2"

                with m.State("READ2"):
                    # read 2, ready for 1
                    m.d.sync += tap[2].eq(rd.data)
                    m.d.comb += [rd.en.eq(1)]
                    m.next = "READ1"

                with m.State("READ1"):
                    # read 1, ready for 0
                    m.d.sync += tap[1].eq(rd.data)
                    m.d.comb += [rd.en.eq(1)]
                    m.next = "OUT_PREP"

                with m.State("OUT_PREP"):
                    # read 02
                    m.d.sync += tap[0].eq(rd.data)
                    m.next = "OUTPUT"

                with m.State("OUTPUT"):
                    # output valid and ready for idle
                    with m.If(self.o.ready):
                        m.next = "IDLE"
                    m.d.comb += [
                        self.o.valid.eq(1),
                    ]

            with m.If(fsm.ongoing("READ3")):
                m.d.comb += rd.addr.eq(idx_accepted[2])
            with m.Elif(fsm.ongoing("READ2")):
                m.d.comb += rd.addr.eq(idx_accepted[1])
            with m.Elif(fsm.ongoing("READ1")):
                m.d.comb += rd.addr.eq(idx_accepted[0])

        else:
            ff_read_head = Signal(range(n), init=0)
            ff_idx = Array(Signal(range(n), name=f"ff_idx_{i}") for i in range(3))

            m.d.comb += [
                ff_read_head.eq(
                    Mux(
                        self.i.valid & self.i.ready,
                        (self.write_head - 1) & ring_mask,
                        self.write_head,
                    )
                ),
                ff_idx[0].eq((ff_read_head - d) & ring_mask),
                ff_idx[1].eq((ff_read_head - (2 * d)) & ring_mask),
                ff_idx[2].eq((ff_read_head - (3 * d)) & ring_mask),
                self.i.ready.eq(self.o.ready),
                self.o.valid.eq(self.i.valid),
                self.o.payload[0].eq(self.buffer[ff_idx[2]]),
                self.o.payload[1].eq(self.buffer[ff_idx[1]]),
                self.o.payload[2].eq(self.buffer[ff_idx[0]]),
                self.o.payload[3].eq(self.i.payload),
            ]

            with m.If(self.i.valid & self.i.ready):
                m.d.sync += self.buffer[self.write_head].eq(self.i.payload)

        with m.If(self.i.valid & self.i.ready):
            m.d.sync += self.write_head.eq((self.write_head + 1) & ring_mask)

        return m
