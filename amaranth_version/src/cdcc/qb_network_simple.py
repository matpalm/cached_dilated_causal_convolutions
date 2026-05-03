import pickle

from amaranth import Module
from amaranth.lib import data, stream, wiring
import numpy as np
from numpy.typing import NDArray

from . import K, NNQ
from .conv1d import Conv1d
from .left_shift_buffer import LeftShiftBuffer

# TODO: for now this is just a single conv1d, no relu and no


def build_network(weights_pkl: str):
    with open(weights_pkl, "rb") as f:
        d = pickle.load(f)
        weights, biases = d["qconv_0_qb"]["weights"]
        return QbNetworkSimple(weights, biases)


class QbNetworkSimple(wiring.Component):

    IN_D = 4
    OUT_D = 4

    def __init__(self, np_weights: NDArray, np_biases: NDArray):
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

        if len(np_biases.shape) != 1:
            raise Exception(
                "Expect bias shape (OUT_D,) " f"but received {np_bias.shape}"
            )

        if np_biases.shape[0] != self.OUT_D:
            raise Exception(
                f"Expect bias shape ({self.OUT_D},) " f"but received {np_bias.shape}"
            )

        self._lsb = LeftShiftBuffer()
        self._conv = Conv1d(np_weights, np_biases, apply_relu=False)

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

        conv_o0 = stream.Signature(NNQ).create()

        wiring.connect(m, wiring.flipped(self.i), self._lsb.i)
        wiring.connect(m, self._lsb.o, self._conv.i)

        m.d.comb += [
            conv_o0.valid.eq(self._conv.o.valid),
            self._conv.o.ready.eq(conv_o0.ready),
            conv_o0.payload.eq(self._conv.o.payload[0]),
        ]

        wiring.connect(m, conv_o0, wiring.flipped(self.o))

        return m
