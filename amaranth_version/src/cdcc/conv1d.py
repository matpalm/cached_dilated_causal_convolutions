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
        np_bias: NDArray,
        apply_relu: bool = False,
    ):
        if len(np_weights.shape) != 3:
            raise Exception(
                "Expect Conv1d weights with shape (NUM_KERNELS, OUT_D, IN_D) "
                f"but received {np_weights.shape}"
            )

        num_kernels, self.OUT_D, self.IN_D = np_weights.shape

        if num_kernels != K:
            raise Exception(
                f"Expect Conv1d weights first axis to be {K} but received {np_weights.shape[0]}"
            )

        if len(np_bias.shape) != 1 or np_bias.shape[0] != self.OUT_D:
            raise Exception(
                f"Expect Conv1d bias with shape ({self.OUT_D},) "
                f"but received {np_bias.shape}"
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

        self._kernels = [RowByMatrixMultiply(np_weights[k]) for k in range(K)]
        self._bias = Array(parse_nnq(np_bias))
        self._apply_relu = apply_relu

        self._accum = Array(
            Signal(NNQ_DW, name=f"conv_accum_{i}", init=0) for i in range(self.OUT_D)
        )
        self._result = Array(
            Signal(NNQ, name=f"conv_result_{i}", init=0) for i in range(self.OUT_D)
        )

        # the max value for NNQ single precision is 7.999755859375 whereas the min value is -8
        # so to avoid overflow we clip the double width precision
        # value between these bounds _before_ the single precision conversion
        self._lower_bound = fixed.Const(-8.0, shape=NNQ_DW).as_value()
        self._upper_bound = fixed.Const(7.999755859375, shape=NNQ_DW).as_value()

    def elaborate(self, platform):
        m = Module()

        for i in range(self.OUT_D):
            m.d.comb += self.o.payload[i].eq(self._result[i])

        all_kernels_ready = 1
        all_kernels_valid = 1
        accept_inputs = Signal(init=0)

        for k, kernel in enumerate(self._kernels):
            m.submodules[f"kernel{k}"] = kernel

            m.d.comb += [
                kernel.i.payload.eq(self.i.payload[k]),
                kernel.i.valid.eq(self.i.valid & accept_inputs),
                kernel.o.ready.eq(1),
            ]

            all_kernels_ready = all_kernels_ready & kernel.i.ready
            all_kernels_valid = all_kernels_valid & kernel.o.valid

        with m.FSM():

            # TODO: do we need an IDLE state here? or is it just latency?

            with m.State("MAT_MUL_RUNNING"):
                m.d.comb += [
                    accept_inputs.eq(1),
                    self.i.ready.eq(all_kernels_ready),
                ]

                with m.If(all_kernels_valid):
                    for i in range(self.OUT_D):
                        m.d.sync += self._accum[i].eq(
                            self._kernels[0].o.payload[i]
                            + self._kernels[1].o.payload[i]
                            + self._kernels[2].o.payload[i]
                            + self._kernels[3].o.payload[i]
                            + self._bias[i]
                        )
                    m.next = "CLIP_LOWER"

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
                    # TODO: this slicing 12:28 depends on the the sizing of NNQ and NNQ_DW
                    #       is there a better way to more directly do this conversion?
                    m.d.sync += self._result[i].eq(
                        self._accum[i].as_value()[12:28].as_signed()
                    )
                if self._apply_relu:
                    m.next = "APPLY_RELU"
                else:
                    m.next = "OUTPUT"

            with m.State("APPLY_RELU"):
                for i in range(self.OUT_D):
                    m.d.sync += self._result[i].eq(
                        Mux(
                            self._result[i].as_value()[-1],
                            0,
                            self._result[i],
                        )
                    )
                m.next = "OUTPUT"

            with m.State("OUTPUT"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "MAT_MUL_RUNNING"

        return m
