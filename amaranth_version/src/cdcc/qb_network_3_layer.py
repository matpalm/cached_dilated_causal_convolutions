import pickle

from amaranth import Module
from amaranth.lib import data, stream, wiring
import numpy as np
from numpy.typing import NDArray

from . import K, NNQ
from .conv1d import Conv1d
from .left_shift_buffer import LeftShiftBuffer
from .activation_cache import ActivationCache
from .stream_cut import StreamCut

class QbNetworkThreeLayer(wiring.Component):

    @staticmethod
    def build(weights_pkl: str):
        with open(weights_pkl, "rb") as f:
            data = pickle.load(f)
            return QbNetworkThreeLayer(data)

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
        return w, b

    def elaborate(self, platform):
        m = Module()

        m.submodules.lsb = lsb = LeftShiftBuffer()

        w, b = self.conv_weights_biases_for("qconv_0_qb")
        m.submodules.conv0 = conv0 = Conv1d(w, b, apply_relu=True)

        num_filters = len(b)
        m.submodules.act0 = act0 = ActivationCache(
            in_out_d=num_filters, dilation_level=1, use_ebr=False
        )

        w, b = self.conv_weights_biases_for("qconv_1_qb")
        m.submodules.conv1 = conv1 = Conv1d(w, b, apply_relu=True)

        num_filters = len(b)
        m.submodules.act1 = act1 = ActivationCache(
            in_out_d=num_filters, dilation_level=2, use_ebr=True
        )

        w, b = self.conv_weights_biases_for("qconv_2_qb")
        m.submodules.conv2 = conv2 = Conv1d(w, b, apply_relu=False)

        # inject cuts between convs and activation caches
        # drops TRELLIS_COMB slightly, but ups TRELLIS_FF
        # most importantly makes routing a lot faster
        m.submodules.cut_conv0_to_act0 = cut_conv0_to_act0 = StreamCut(
            data.ArrayLayout(NNQ, conv0.OUT_D)
        )
        m.submodules.cut_act0_to_conv1 = cut_act0_to_conv1 = StreamCut(
            data.ArrayLayout(data.ArrayLayout(NNQ, conv1.IN_D), K)
        )
        m.submodules.cut_conv1_to_act1 = cut_conv1_to_act1 = StreamCut(
            data.ArrayLayout(NNQ, conv1.OUT_D)
        )
        m.submodules.cut_act1_to_conv2 = cut_act1_to_conv2 = StreamCut(
            data.ArrayLayout(data.ArrayLayout(NNQ, conv2.IN_D), K)
        )

        waveshaped_output = stream.Signature(NNQ).create()

        wiring.connect(m, wiring.flipped(self.i), lsb.i)
        wiring.connect(m, lsb.o, conv0.i)
        wiring.connect(m, conv0.o, cut_conv0_to_act0.i)
        wiring.connect(m, cut_conv0_to_act0.o, act0.i)
        wiring.connect(m, act0.o, cut_act0_to_conv1.i)
        wiring.connect(m, cut_act0_to_conv1.o, conv1.i)
        wiring.connect(m, conv1.o, cut_conv1_to_act1.i)
        wiring.connect(m, cut_conv1_to_act1.o, act1.i)
        wiring.connect(m, act1.o, cut_act1_to_conv2.i)
        wiring.connect(m, cut_act1_to_conv2.o, conv2.i)

        final_conv = conv2
        m.d.comb += [
            waveshaped_output.valid.eq(final_conv.o.valid),
            final_conv.o.ready.eq(waveshaped_output.ready),
            waveshaped_output.payload.eq(final_conv.o.payload[0]),
        ]

        wiring.connect(m, waveshaped_output, wiring.flipped(self.o))

        return m
