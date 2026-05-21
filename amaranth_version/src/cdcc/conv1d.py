from amaranth import Array, Module, Mux, Signal
from amaranth.lib import data, stream, wiring
import numpy as np
from numpy.typing import NDArray
from amaranth_future import fixed

from . import NNQ, NNQ_DW, K, parse_nnq
from .row_by_matrix_multiply import RowByMatrixMultiply

class Conv1d(wiring.Component):

    def __init__(
        self,
        np_weights: NDArray,
        np_biases: NDArray,
        apply_relu: bool,
        relu_upper_bound: int = 6,
    ):
        """
        Args:
            np_weights  (K=4, IN_D, OUT_D)
            np_bias     (OUT_D)
        """

        if len(np_weights.shape) != 3:
            raise Exception(
                "Expect Conv1d weights with shape (NUM_KERNELS, IN_D, OUT_D) "
                f"but received {np_weights.shape}"
            )

        num_kernels, self.IN_D, self.OUT_D = np_weights.shape

        if num_kernels != K:
            raise Exception(
                f"Expect Conv1d weights first axis to be {K} but received {np_weights.shape[0]}"
            )

        if len(np_biases.shape) != 1 or np_biases.shape[0] != self.OUT_D:
            raise Exception(
                f"Expect Conv1d bias with shape ({self.OUT_D},) "
                f"but received {np_biases.shape}"
            )

        super().__init__(
            {
                "i": wiring.In(
                    stream.Signature(
                        data.ArrayLayout(data.ArrayLayout(NNQ, self.IN_D), K)
                    )
                ),
                "o": wiring.Out(stream.Signature(data.ArrayLayout(NNQ, self.OUT_D))),
            }
        )

        self.relu_upper_bound = fixed.Const(relu_upper_bound, shape=NNQ)

        self.row_mults = [
            RowByMatrixMultiply(np_weights[0], np_weights_alt=np_weights[2]),
            RowByMatrixMultiply(np_weights[1], np_weights_alt=np_weights[3]),
        ]
        self.biases = Array(parse_nnq(b, shape=NNQ_DW) for b in np_biases)
        self.apply_relu = apply_relu

        self.accum = Array(
            Signal(NNQ_DW, name=f"conv_accum_{i}", init=0) for i in range(self.OUT_D)
        )
        self.result = Array(
            Signal(NNQ, name=f"conv_result_{i}", init=0) for i in range(self.OUT_D)
        )
        self.input = Array(
            Array(
                Signal(NNQ, name=f"conv_in_{k}_{i}", init=0) for i in range(self.IN_D)
            )
            for k in range(K)
        )

        # clip to representable NNQ bounds (expressed in NNQ_DW shape)
        # before narrowing NNQ_DW -> NNQ. note: can't use fixed Value utils (clamp)
        # directly since we're matching FXPmath/qkeras which does things slightly
        # differently
        self.lower_bound = fixed.Const(NNQ.min().as_float(), shape=NNQ_DW).as_value()
        self.upper_bound = fixed.Const(NNQ.max().as_float(), shape=NNQ_DW).as_value()

    def elaborate(self, platform):
        m = Module()

        m.d.comb += [
            self.i.ready.eq(0),
            self.o.valid.eq(0),
        ]

        for i in range(self.OUT_D):
            m.d.comb += self.o.payload[i].eq(self.result[i])

        with m.FSM() as fsm:

            all_rows_ready = 1
            all_rows_valid = 1
            phase1 = fsm.ongoing("START_PHASE1") | fsm.ongoing("WAIT_PHASE1")

            for k, rbmm in enumerate(self.row_mults):
                m.submodules[f"row_mm_{k}"] = rbmm

                m.d.comb += [
                    rbmm.phase.eq(phase1),
                    rbmm.i.valid.eq(
                        fsm.ongoing("START_PHASE0") | fsm.ongoing("START_PHASE1")
                    ),
                    rbmm.o.ready.eq(
                        fsm.ongoing("WAIT_PHASE0") | fsm.ongoing("WAIT_PHASE1")
                    ),
                ]
                for i in range(self.IN_D):
                    m.d.comb += rbmm.i.payload[i].eq(
                        Mux(phase1, self.input[k + 2][i], self.input[k][i])
                    )

                all_rows_ready = all_rows_ready & rbmm.i.ready
                all_rows_valid = all_rows_valid & rbmm.o.valid

            frac_drop = NNQ_DW.f_bits - NNQ.f_bits
            out_width = NNQ.width

            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(all_rows_ready)
                with m.If(self.i.valid & self.i.ready):
                    for k in range(K):
                        for i in range(self.IN_D):
                            m.d.sync += self.input[k][i].eq(self.i.payload[k][i])
                    m.next = "START_PHASE0"

            with m.State("START_PHASE0"):
                with m.If(all_rows_ready):
                    m.next = "WAIT_PHASE0"

            with m.State("WAIT_PHASE0"):
                with m.If(all_rows_valid):
                    for i in range(self.OUT_D):
                        row_sum = self.row_mults[0].o.payload[i].as_value().as_signed()
                        for k in range(1, 2):
                            row_sum = (
                                row_sum
                                + self.row_mults[k].o.payload[i].as_value().as_signed()
                            )
                        m.d.sync += self.accum[i].eq(
                            self.biases[i].as_value().as_signed() + row_sum
                        )
                    m.next = "START_PHASE1"

            with m.State("START_PHASE1"):
                with m.If(all_rows_ready):
                    m.next = "WAIT_PHASE1"

            with m.State("WAIT_PHASE1"):
                with m.If(all_rows_valid):
                    for i in range(self.OUT_D):
                        row_sum = self.row_mults[0].o.payload[i].as_value().as_signed()
                        for k in range(1, 2):
                            row_sum = (
                                row_sum
                                + self.row_mults[k].o.payload[i].as_value().as_signed()
                            )
                        m.d.sync += self.accum[i].eq(
                            self.accum[i].as_value().as_signed() + row_sum
                        )
                    m.next = "CLIP_LOWER"

            with m.State("CLIP_LOWER"):
                # TODO: combine CLIP_LOWER and _UPPER?
                for i in range(self.OUT_D):
                    m.d.sync += self.accum[i].eq(
                        Mux(
                            self.accum[i] < self.lower_bound,
                            self.lower_bound,
                            self.accum[i],
                        )
                    )
                m.next = "CLIP_UPPER"

            with m.State("CLIP_UPPER"):
                for i in range(self.OUT_D):
                    m.d.sync += self.accum[i].eq(
                        Mux(
                            self.accum[i] > self.upper_bound,
                            self.upper_bound,
                            self.accum[i],
                        )
                    )
                m.next = "SINGLE_W"

            with m.State("SINGLE_W"):
                for i in range(self.OUT_D):
                    # TODO: had to include this because of a weird diff with narrowing
                    # to match fxpmath :/ ( specifically difference in truncate toward zero
                    # behaviour )
                    acc = self.accum[i].as_value()
                    acc_clipped = Mux(
                        acc < self.lower_bound,
                        self.lower_bound,
                        Mux(acc > self.upper_bound, self.upper_bound, acc),
                    )
                    frac_nonzero = acc_clipped[:frac_drop].any()
                    trunc_toward_zero = Mux(
                        acc_clipped[-1] & frac_nonzero,
                        acc_clipped + (1 << frac_drop),
                        acc_clipped,
                    )
                    m.d.sync += self.result[i].eq(
                        trunc_toward_zero[frac_drop : frac_drop + out_width].as_signed()
                    )
                if self.apply_relu:
                    m.next = "APPLY_RELU"
                else:
                    m.next = "OUTPUT"

            with m.State("APPLY_RELU"):
                for i in range(self.OUT_D):
                    m.d.sync += self.result[i].eq(
                        Mux(
                            self.result[i].as_value()[-1],
                            0,
                            Mux(
                                self.result[i] > self.relu_upper_bound,
                                self.relu_upper_bound,
                                self.result[i],
                            ),
                        )
                    )
                m.next = "OUTPUT"

            with m.State("OUTPUT"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
