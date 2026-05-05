import pickle

from amaranth import Module
from amaranth.lib import data, stream, wiring

from . import NNQ
from .conv1d import Conv1d
from .left_shift_buffer import LeftShiftBuffer

class QbNetworkOneLayer(wiring.Component):

    @staticmethod
    def build(weights_pkl: str):
        with open(weights_pkl, "rb") as f:
            data = pickle.load(f)
            return QbNetworkOneLayer(data)

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
        m.submodules.conv0 = conv0 = Conv1d(w, b, apply_relu=False)

        waveshaped_output = stream.Signature(NNQ).create()

        wiring.connect(m, wiring.flipped(self.i), lsb.i)
        wiring.connect(m, lsb.o, conv0.i)

        final_conv = conv0
        m.d.comb += [
            waveshaped_output.valid.eq(final_conv.o.valid),
            final_conv.o.ready.eq(waveshaped_output.ready),
            waveshaped_output.payload.eq(final_conv.o.payload[0]),
        ]

        wiring.connect(m, waveshaped_output, wiring.flipped(self.o))

        return m
