import pickle

from amaranth import Module
from amaranth.lib import data, stream, wiring
import numpy as np
from numpy.typing import NDArray

from . import K, NNQ
from .conv1d import Conv1d
from .left_shift_buffer import LeftShiftBuffer
from .activation_cache import ActivationCache
from .stream_select_lane import StreamSelectLane
from .stream_register import StreamRegister

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
        print(conv_name, "w", w.shape, "b", b.shape)
        return w, b

    def elaborate(self, platform):
        m = Module()

        m.submodules.lsb = lsb = LeftShiftBuffer()

        w, b = self.conv_weights_biases_for("qconv_0_qb")
        m.submodules.conv0 = conv0 = Conv1d(w, b, apply_relu=False)
        m.submodules.select_lane = select_lane = StreamSelectLane(
            in_d=conv0.OUT_D, lane_index=0
        )
        m.submodules.output_reg = output_reg = StreamRegister(NNQ)

        wiring.connect(m, wiring.flipped(self.i), lsb.i)
        wiring.connect(m, lsb.o, conv0.i)
        wiring.connect(m, conv0.o, select_lane.i)
        wiring.connect(m, select_lane.o, output_reg.i)
        wiring.connect(m, output_reg.o, wiring.flipped(self.o))

        return m
