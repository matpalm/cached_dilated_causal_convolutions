from amaranth import Array, Module, Mux, Signal
from amaranth.lib import data, stream, wiring
import numpy as np
from numpy.typing import NDArray
from amaranth_future import fixed

from . import NNQ, NNQ_DW, K, parse_nnq

class Conv1d(wiring.Component):

    def __init__(
        self,
        np_weights: NDArray,
        np_biases: NDArray,
        apply_relu: bool,
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

        # we don't have enough MULT18X18D units anymore to run all kernels in parallel
        # so instead run the K=4 sequentially.

        self.weights = Array(
            Array(Array(parse_nnq(np_weights[k, i])) for i in range(self.IN_D))
            for k in range(K)
        )
        self.biases = Array(parse_nnq(np_biases, shape=NNQ_DW))
        self.apply_relu = apply_relu

        self.k_idx = Signal(range(K), init=0)
        self.i_idx = Signal(range(self.IN_D), init=0)
        self.o_idx = Signal(range(self.OUT_D), init=0)

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

        with m.FSM():

            frac_drop = NNQ_DW.f_bits - NNQ.f_bits
            out_width = NNQ.width

            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid & self.i.ready):
                    for k in range(K):
                        for i in range(self.IN_D):
                            m.d.sync += self.input[k][i].eq(self.i.payload[k][i])
                    for i in range(self.OUT_D):
                        m.d.sync += self.accum[i].eq(self.biases[i])
                    m.d.sync += [
                        self.k_idx.eq(0),
                        self.i_idx.eq(0),
                        self.o_idx.eq(0),
                    ]
                    m.next = "MAT_MUL_RUNNING"

            with m.State("MAT_MUL_RUNNING"):
                m.d.sync += self.accum[self.o_idx].eq(
                    self.accum[self.o_idx].as_value().as_signed()
                    + (
                        self.input[self.k_idx][self.i_idx].as_value().as_signed()
                        * self.weights[self.k_idx][self.i_idx][self.o_idx]
                        .as_value()
                        .as_signed()
                    )
                )

                with m.If(self.i_idx == self.IN_D - 1):
                    m.d.sync += self.i_idx.eq(0)
                    with m.If(self.o_idx == self.OUT_D - 1):
                        m.d.sync += self.o_idx.eq(0)
                        with m.If(self.k_idx == K - 1):
                            m.next = "CLIP_LOWER"
                        with m.Else():
                            m.d.sync += self.k_idx.eq(self.k_idx + 1)
                    with m.Else():
                        m.d.sync += self.o_idx.eq(self.o_idx + 1)
                with m.Else():
                    m.d.sync += self.i_idx.eq(self.i_idx + 1)

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
                    # Ensure saturating behavior during narrowing, matching
                    # fxpmath resize semantics used by the reference model.
                    # Match fxpmath resize semantics (truncate toward zero)
                    # while narrowing NNQ_DW -> NNQ using shape-derived widths.
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
                    m.next = "APPLY_RELU_6"
                else:
                    m.next = "OUTPUT"

            with m.State("APPLY_RELU_6"):
                for i in range(self.OUT_D):
                    m.d.sync += self.result[i].eq(
                        Mux(
                            self.result[i].as_value()[-1],
                            0,
                            Mux(
                                self.result[i] > fixed.Const(6.0, shape=NNQ),
                                fixed.Const(6.0, shape=NNQ),
                                self.result[i],
                            ),
                        )
                    )
                m.next = "OUTPUT"

            with m.State("OUTPUT"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    # Single in-flight transaction: do not accept next input
                    # until the current output beat has been consumed.
                    m.next = "IDLE"

        return m
