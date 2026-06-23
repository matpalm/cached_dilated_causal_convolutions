from amaranth import Array, Cat, Module, Mux, Signal
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import Memory

from . import NNQ, K


class ActivationCache(wiring.Component):
    # note: for dilation=1 this is equivalent to left_shift_buffer

    def __init__(self, in_out_d: int, dilation_level: int):
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

        # note: only need caching back to the furthest tap (3*dilation) plus the
        # current entry => 3*dilation + 1. we _dont_ need K*dilation = 4*dilation entries
        self.num_entries = 3 * self.dilation + 1

        print(
            f">ActivationCache in_out_d={in_out_d}"
            f" dilation_level={dilation_level} => dilation={self.dilation} => |entries|={self.num_entries}"
        )

        feature = self.input_layout
        self.buffer = None
        self.ebr_memory = None

        init = [[0] * self.in_out_d for _ in range(self.num_entries)]
        self.ebr_memory = Memory(
            shape=feature,
            depth=self.num_entries,
            init=init,
            attrs={"ram_style": "block"},
        )
        self.write_head = Signal(range(self.num_entries), init=0)

    def elaborate(self, platform):
        m = Module()

        d = self.dilation
        n = self.num_entries

        # helper for ring based indexing
        def sub_mod(a, b):
            return Mux(a >= b, a - b, a - b + n)

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
            idx_accepted[0].eq(sub_mod(accepted_head, d)),
            idx_accepted[1].eq(sub_mod(accepted_head, 2 * d)),
            idx_accepted[2].eq(sub_mod(accepted_head, 3 * d)),
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

        with m.If(self.i.valid & self.i.ready):
            m.d.sync += self.write_head.eq(
                Mux(self.write_head == n - 1, 0, self.write_head + 1)
            )

        return m
