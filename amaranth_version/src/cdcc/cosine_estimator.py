import math

from amaranth.lib import stream, wiring, data
from amaranth.lib.memory import Memory
from amaranth import Module, Signal, Mux

from . import NNQ


class CosineEstimator(wiring.Component):
    # given a incoming sine wave estimate the corresponding cosine wave
    # by checking zero crossings to derive frequency and from that look back
    # in a a delay line of the sine values. won't work for sine wave under heavy FM

    # incoming sine wave
    i: wiring.In(stream.Signature(NNQ))

    # derived cosine
    o: wiring.Out(stream.Signature(NNQ))

    def __init__(self, decimation=64):
        super().__init__()

    def elaborate(self, platform):
        m = Module()

        # circular buffer for sine values
        # 256 should be ok for 192kHz sines across ranges (?)
        N = 256
        circular_buffer = Memory(
            shape=NNQ,
            depth=N,
            init=[0] * N,
            attrs={"ram_style": "block"},
        )
        m.submodules.circular_buffer = circular_buffer
        rd = circular_buffer.read_port(domain="sync")
        wr = circular_buffer.write_port(domain="sync")

        # size of indexs to track sample ids
        # something weird will happen at idx wrapping (?)
        IDX_SIZE = 32

        # accepted and valid_d ( accepted delayed by 1)
        accepted = Signal()
        valid_d = Signal()

        # read / write and estimate heads
        write_head = Signal(range(N))
        quarter_period = Signal(range(N), init=N // 2)  # stay away from 0 at start
        cosine_offset_estimate = Signal(IDX_SIZE)

        # tracking of cross -ve to +ve, including idxs for crossing and calculating
        # samples since last crossing.
        is_negative = Signal(init=0)
        is_negative_d = Signal(init=0)
        crossing_zero = Signal()
        current_sample_idx = Signal(IDX_SIZE, init=0)
        last_cross_idx = Signal(IDX_SIZE, init=0)
        samples_since_last_cross = Signal(IDX_SIZE, init=0)
        # note: need to allow at least one to get valid value for last crossing
        at_least_one_cross = Signal(init=0)

        m.d.comb += [
            accepted.eq(self.i.valid & self.i.ready),
            is_negative.eq(self.i.payload.as_value()[-1]),
            crossing_zero.eq(accepted & is_negative_d & ~is_negative),
            samples_since_last_cross.eq(current_sample_idx - last_cross_idx),
            cosine_offset_estimate.eq(samples_since_last_cross >> 2),
            # sync read returns data one cycle later, so advance address by one
            # sample to compensate that pipeline latency (?)
            rd.addr.eq((write_head - quarter_period + 1) & 0xFF),
            wr.addr.eq(write_head),
            wr.data.eq(self.i.payload),
            wr.en.eq(accepted),
            # self.o.payload[0].eq(aligned_sin),
            self.o.payload.eq(-rd.data),  # note: cosine is -ve delayed sine
            self.o.valid.eq(valid_d),
            self.i.ready.eq(1),
        ]

        with m.If(accepted):
            m.d.sync += [
                write_head.eq(write_head + 1),
                current_sample_idx.eq(current_sample_idx + 1),
                is_negative_d.eq(is_negative),
                # aligned_sin.eq(self.i.payload),
            ]

        with m.If(crossing_zero):
            with m.If(at_least_one_cross):
                m.d.sync += quarter_period.eq(
                    Mux(
                        cosine_offset_estimate < 1,
                        1,
                        Mux(
                            cosine_offset_estimate > N - 1,
                            N - 1,
                            cosine_offset_estimate,
                        ),
                    )
                )
            m.d.sync += [
                last_cross_idx.eq(current_sample_idx),
                at_least_one_cross.eq(1),
            ]

        m.d.sync += valid_d.eq(accepted)

        return m
