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

        self._weights = Array(
            Array(Array(parse_nnq(np_weights[k, i])) for i in range(self.IN_D))
            for k in range(K)
        )
        self._biases = Array(parse_nnq(np_biases, shape=NNQ_DW))
        self._apply_relu = apply_relu

        self._k_idx = Signal(range(K), init=0)
        self._i_idx = Signal(range(self.IN_D), init=0)
        self._o_idx = Signal(range(self.OUT_D), init=0)

        self._accum = Array(
            Signal(NNQ_DW, name=f"conv_accum_{i}", init=0) for i in range(self.OUT_D)
        )
        self._result = Array(
            Signal(NNQ, name=f"conv_result_{i}", init=0) for i in range(self.OUT_D)
        )
        self._input = Array(
            Array(
                Signal(NNQ, name=f"conv_in_{k}_{i}", init=0) for i in range(self.IN_D)
            )
            for k in range(K)
        )

        # the max value for NNQ single precision is 7.999755859375 whereas the min value is -8
        # so to avoid overflow we clip the double width precision
        # value between these bounds _before_ the single precision conversion
        self._lower_bound = fixed.Const(-8.0, shape=NNQ_DW).as_value()
        self._upper_bound = fixed.Const(7.999755859375, shape=NNQ_DW).as_value()

    def elaborate(self, platform):
        m = Module()

        m.d.comb += [
            self.i.ready.eq(0),
            self.o.valid.eq(0),
        ]

        for i in range(self.OUT_D):
            m.d.comb += self.o.payload[i].eq(self._result[i])

        with m.FSM():

            frac_drop = NNQ_DW.f_bits - NNQ.f_bits
            out_width = NNQ.width

            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid & self.i.ready):
                    for k in range(K):
                        for i in range(self.IN_D):
                            m.d.sync += self._input[k][i].eq(self.i.payload[k][i])
                    for i in range(self.OUT_D):
                        m.d.sync += self._accum[i].eq(self._biases[i])
                    m.d.sync += [
                        self._k_idx.eq(0),
                        self._i_idx.eq(0),
                        self._o_idx.eq(0),
                    ]
                    m.next = "MAT_MUL_RUNNING"

            with m.State("MAT_MUL_RUNNING"):
                m.d.sync += self._accum[self._o_idx].eq(
                    self._accum[self._o_idx].as_value().as_signed()
                    + (
                        self._input[self._k_idx][self._i_idx].as_value().as_signed()
                        * self._weights[self._k_idx][self._i_idx][self._o_idx]
                        .as_value()
                        .as_signed()
                    )
                )

                # gawd, what a mess! :/

                with m.If(self._i_idx == self.IN_D - 1):
                    m.d.sync += self._i_idx.eq(0)
                    with m.If(self._o_idx == self.OUT_D - 1):
                        m.d.sync += self._o_idx.eq(0)
                        with m.If(self._k_idx == K - 1):
                            m.next = "CLIP_LOWER"
                        with m.Else():
                            m.d.sync += self._k_idx.eq(self._k_idx + 1)
                    with m.Else():
                        m.d.sync += self._o_idx.eq(self._o_idx + 1)
                with m.Else():
                    m.d.sync += self._i_idx.eq(self._i_idx + 1)

            with m.State("CLIP_LOWER"):
                # TODO: combine CLIP_LOWER and _UPPER?
                for i in range(self.OUT_D):
                    m.d.sync += self._accum[i].eq(
                        Mux(
                            self._accum[i] < self._lower_bound,
                            self._lower_bound,
                            self._accum[i],
                        )
                    )
                m.next = "CLIP_UPPER"

            with m.State("CLIP_UPPER"):
                for i in range(self.OUT_D):
                    m.d.sync += self._accum[i].eq(
                        Mux(
                            self._accum[i] > self._upper_bound,
                            self._upper_bound,
                            self._accum[i],
                        )
                    )
                m.next = "SINGLE_W"

            with m.State("SINGLE_W"):
                for i in range(self.OUT_D):
                    # Ensure saturating behavior during narrowing, matching
                    # fxpmath resize semantics used by the reference model.
                    # Match fxpmath resize semantics (truncate toward zero)
                    # while narrowing NNQ_DW -> NNQ using shape-derived widths.
                    acc = self._accum[i].as_value()
                    acc_clipped = Mux(
                        acc < self._lower_bound,
                        self._lower_bound,
                        Mux(acc > self._upper_bound, self._upper_bound, acc),
                    )
                    frac_nonzero = acc_clipped[:frac_drop].any()
                    trunc_toward_zero = Mux(
                        acc_clipped[-1] & frac_nonzero,
                        acc_clipped + (1 << frac_drop),
                        acc_clipped,
                    )
                    m.d.sync += self._result[i].eq(
                        trunc_toward_zero[frac_drop : frac_drop + out_width].as_signed()
                    )
                if self._apply_relu:
                    m.next = "APPLY_RELU_6"
                else:
                    m.next = "OUTPUT"

            with m.State("APPLY_RELU_6"):
                for i in range(self.OUT_D):
                    m.d.sync += self._result[i].eq(
                        Mux(
                            self._result[i].as_value()[-1],
                            0,
                            Mux(
                                self._result[i] > fixed.Const(6.0, shape=NNQ),
                                fixed.Const(6.0, shape=NNQ),
                                self._result[i],
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
