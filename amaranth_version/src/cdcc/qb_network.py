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


class QbNetwork(wiring.Component):

    @staticmethod
    def build(weights_pkl: str):
        with open(weights_pkl, "rb") as f:
            data = pickle.load(f)
            return QbNetwork(data)

    def __init__(self, qkeras_weights: dict):
        self.qkeras_weights = qkeras_weights
        self.num_layers = len(self.qkeras_weights.keys())
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

        m.submodules["lsb"] = lsb = LeftShiftBuffer(in_out_d=4)

        # build convolutions as well as activation caches
        # for all but the last
        convs = []
        activation_caches = []
        for i in range(self.num_layers):
            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            w, b = self.conv_weights_biases_for(f"qconv_{i}_qb")
            print(f"{i} CONV apply_relu={not last_layer} w {w.shape} b {b.shape}")
            # TODO: hardcoded upper bound here!
            # see https://github.com/matpalm/cached_dilated_causal_convolutions/issues/24
            print("!" * 100, "ASSUMING RELU4")
            conv = Conv1d(w, b, apply_relu=(not last_layer), relu_upper_bound=4.0)
            m.submodules[f"conv{i}"] = conv
            convs.append(conv)

            if not last_layer:
                print(f"{i} ACT CACHE use_ebr=True")
                num_filters = len(b)
                activation_cache = ActivationCache(
                    in_out_d=num_filters,
                    dilation_level=(i + 1),
                    use_ebr=True,
                )
                m.submodules[f"act{i}"] = activation_cache
                activation_caches.append(activation_cache)

        # inject cuts after each activation cache
        # to break long handshake paths.
        # drops TRELLIS_COMB slightly, but ups TRELLIS_FF
        # most importantly makes routing a lot faster
        cut_act_convs = []
        for i in range(self.num_layers - 1):  # ie. NOT last layer
            print(f"{i} cut_act_convs")
            cut_act_conv = StreamCut(activation_caches[i].output_layout)
            m.submodules[f"cut_act_conv{i}"] = cut_act_conv
            cut_act_convs.append(cut_act_conv)

        # do wiring; inp -> left shift -> first conv
        wiring.connect(m, wiring.flipped(self.i), lsb.i)
        wiring.connect(m, lsb.o, convs[0].i)

        # for every layer ( except the last ) wire it up to it's activation
        # cache and then to the next conv ( with each one having cut stream )
        for i in range(self.num_layers - 1):
            wiring.connect(m, convs[i].o, activation_caches[i].i)
            wiring.connect(m, activation_caches[i].o, cut_act_convs[i].i)
            wiring.connect(m, cut_act_convs[i].o, convs[i + 1].i)

        # final connection and hand shaking
        waveshaped_output = stream.Signature(NNQ).create()
        final_conv = convs[-1]
        m.d.comb += [
            waveshaped_output.valid.eq(final_conv.o.valid),
            final_conv.o.ready.eq(waveshaped_output.ready),
            waveshaped_output.payload.eq(final_conv.o.payload[0]),
        ]
        wiring.connect(m, waveshaped_output, wiring.flipped(self.o))

        return m
