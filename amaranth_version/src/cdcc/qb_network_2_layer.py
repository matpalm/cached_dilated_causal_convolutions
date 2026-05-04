import pickle

from amaranth import Module
from amaranth.lib import data, stream, wiring
import numpy as np
from numpy.typing import NDArray

from . import K, NNQ
from .conv1d import Conv1d
from .left_shift_buffer import LeftShiftBuffer
from .activation_cache import ActivationCache


class QbNetworkTwoLayer(wiring.Component):

    @staticmethod
    def build(weights_pkl: str):
        with open(weights_pkl, "rb") as f:
            data = pickle.load(f)
            return QbNetworkTwoLayer(data)

    def __init__(self, qkeras_weights: dict):

        self.qkeras_weights = qkeras_weights
        self.IN_D = 4

        super().__init__(
            {
                "i": wiring.In(stream.Signature(data.ArrayLayout(NNQ, self.IN_D))),
                "o": wiring.Out(stream.Signature(NNQ)),
            }
        )

    def conv_weights_biases_for(self, conv_name: str):
        w, b = self.qkeras_weights[conv_name]["weights"]
        print(conv_name, "w", w.shape, "b", b.shape)
        return w, b

    def elaborate(self, platform):
        m = Module()

        m.submodules.lsb = lsb = LeftShiftBuffer()

        w, b = self.conv_weights_biases_for("qconv_0_qb")
        m.submodules.conv0 = conv0 = Conv1d(w, b, apply_relu=True)

        num_filters = len(b)
        m.submodules.act0 = act0 = ActivationCache(
            in_out_d=num_filters, dilation_level=1
        )

        w, b = self.conv_weights_biases_for("qconv_1_qb")
        m.submodules.conv1 = conv1 = Conv1d(w, b, apply_relu=False)

        if conv0.OUT_D != act0.in_out_d:
            raise ValueError(
                f"conv0 OUT_D ({conv0.OUT_D}) must match act0 in_out_d ({act0.in_out_d})"
            )
        if conv1.IN_D != act0.in_out_d:
            raise ValueError(
                f"conv1 IN_D ({conv1.IN_D}) must match act0 in_out_d ({act0.in_out_d})"
            )

        waveshaped_output = stream.Signature(NNQ).create()

        wiring.connect(m, wiring.flipped(self.i), lsb.i)
        wiring.connect(m, lsb.o, conv0.i)
        wiring.connect(m, conv0.o, act0.i)
        wiring.connect(m, act0.o, conv1.i)

        final_conv = conv1
        m.d.comb += [
            waveshaped_output.valid.eq(final_conv.o.valid),
            final_conv.o.ready.eq(waveshaped_output.ready),
            waveshaped_output.payload.eq(final_conv.o.payload[0]),
        ]

        wiring.connect(m, waveshaped_output, wiring.flipped(self.o))

        return m
