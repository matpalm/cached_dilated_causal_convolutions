from amaranth import Module
from amaranth.lib import data, stream, wiring

import numpy as np
from numpy.typing import NDArray

from . import K, NNQ
from .conv1d import Conv1d
from .left_shift_buffer import LeftShiftBuffer


class QbNetworkSimple(wiring.Component):

    IN_D = 4
    OUT_D = 1

    def __init__(self, np_weights: NDArray, apply_relu: bool = True):
        if len(np_weights.shape) != 3:
            raise Exception(
                "Expect weights shape (K, OUT_D, IN_D) "
                f"but received {np_weights.shape}"
            )

        if np_weights.shape != (K, self.OUT_D, self.IN_D):
            raise Exception(
                f"Expect weights shape {(K, self.OUT_D, self.IN_D)} "
                f"but received {np_weights.shape}"
            )

        self._np_weights = np_weights

        self._lsb = LeftShiftBuffer()
        self._conv = Conv1d(np_weights, apply_relu=apply_relu)

        super().__init__(
            {
                "i": wiring.In(stream.Signature(data.ArrayLayout(NNQ, self.IN_D))),
                "o": wiring.Out(stream.Signature(NNQ)),
            }
        )

    def elaborate(self, platform):
        m = Module()

        m.submodules.lsb = self._lsb
        m.submodules.conv = self._conv

        m.d.comb += [
            self._lsb.i.payload.eq(self.i.payload),
            self._lsb.i.valid.eq(self.i.valid),
            self.i.ready.eq(self._lsb.i.ready),
            self._lsb.o.ready.eq(self._conv.i.ready),
            self._conv.i.payload.eq(self._lsb.o.payload),
            self._conv.i.valid.eq(self._lsb.o.valid),
            self._conv.o.ready.eq(self.o.ready),
            self.o.valid.eq(self._conv.o.valid),
            self.o.payload.eq(self._conv.o.payload[0]),
        ]

        return m
