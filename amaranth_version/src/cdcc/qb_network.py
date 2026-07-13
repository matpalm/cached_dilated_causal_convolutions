import os
import pickle
from ast import literal_eval

from amaranth import Module
from amaranth.lib import data, stream, wiring
import numpy as np
from numpy.typing import NDArray

from . import K, NNQ
from .conv1d import Conv1d
from .left_shift_buffer import LeftShiftBuffer
from .activation_cache import ActivationCache
from .activation_cache_ps import ActivationCachePS
from .stream_cut import StreamCut

# indexes of activation caches to use PSRAM ( the rest stay in EBR ).
# supports negative indexing ( -1 => deepest cache ).
PSRAM_ACTIVATION_CACHE_INDICES = literal_eval(
    os.getenv("PSRAM_ACTIVATION_CACHE_INDICES", "[]")
)
print("PSRAM_ACTIVATION_CACHE_INDICES", PSRAM_ACTIVATION_CACHE_INDICES)

# amount to align for when we have more than one PSRAM cache
# TODO: can't wishbone do this for us?
PSRAM_REGION_ALIGN = 4096


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

        # create the activation caches pre elobration so ....
        # 1) tests can get .bus ports before creating FakePSRAM and
        # 2) qb_network can expose each bus as out ports for "real" psram
        # TODO: 1) feels clunky :/
        #
        # note: the last cache may be PSRAM backed ( wishbone ) while every
        # shallower cache stays in EBR. psram_cache_indices records which
        # caches expose a wishbone bus so qb_network / CoreTop can wire them.
        self.activation_caches = []
        self.psram_cache_indices = []
        num_caches = self.num_layers - 1

        # normalise (potentially -ve) idxs; e.g. -1 => last
        psram_indices = sorted({i % num_caches for i in PSRAM_ACTIVATION_CACHE_INDICES})

        # running byte offset for laying out PSRAM backed caches.
        psram_base = 0

        for i in range(num_caches):
            _, b = self.conv_weights_biases_for(i)
            num_filters = len(b)
            use_psram = i in psram_indices
            kind = "PSRAM" if use_psram else "EBR"
            print(f"{i} ACT CACHE ({kind}) in_out_d={num_filters} dilation_level={i+1}")
            if use_psram:
                cache = ActivationCachePS(
                    in_out_d=num_filters,
                    dilation_level=(i + 1),
                    base=psram_base,
                )
                self.psram_cache_indices.append(i)
                # advance the base past this cache's region. each entry-major
                # 16bit word pairs into a 32bit PSRAM word ( x2 ), so the byte
                # span is ceil(total_words / 2) * 4, aligned upwards.
                # TODO: PSRAM_REGION_ALIGN isn't overly wasteful?
                total_words = cache.num_entries * cache.dim_stride
                byte_span = ((total_words + 1) // 2) * 4
                psram_base += -(-byte_span // PSRAM_REGION_ALIGN) * PSRAM_REGION_ALIGN
            else:
                cache = ActivationCache(
                    in_out_d=num_filters,
                    dilation_level=(i + 1),
                )
            self.activation_caches.append(cache)

        # in / out ports are fixed...
        ports = {
            "i": wiring.In(stream.Signature(data.ArrayLayout(NNQ, self.IN_D))),
            "o": wiring.Out(stream.Signature(NNQ)),
        }
        # ... but a wishbone port per PSRAM backed cache must be made dynamically.
        # TODO: forcing fixed size layers would make this _much_ simpler ( and
        #       is going to be required anyway for hot swapping layers? )
        for i in self.psram_cache_indices:
            ports[f"bus_act{i}"] = wiring.Out(self.activation_caches[i].bus_signature)

        super().__init__(ports)

    def conv_name_for_layer(self, layer_idx: int) -> str:
        # the final layer is the 1x1 projection, exported as "qconv_regressor_qb";
        # every earlier ( K=4 ) layer is "qconv_{i}_qb".
        if layer_idx == self.num_layers - 1:
            return "qconv_regressor_qb"
        return f"qconv_{layer_idx}_qb"

    def conv_weights_biases_for(self, layer_idx: int):
        conv_name = self.conv_name_for_layer(layer_idx)
        w, b = self.qkeras_weights[conv_name]["weights"]
        w = np.asarray(w)
        # the regressor is a kernel_size=1 conv, which we'll also use for later
        # for the skip conenctions. for now, just to make it work, pad the kernel
        # to K=4 to make it work with the conv
        # TODO: support K=1 for not just this but the skips too
        if layer_idx == self.num_layers - 1 and w.shape[0] == 1 and K > 1:
            padded = np.zeros((K,) + w.shape[1:], dtype=w.dtype)
            padded[K - 1] = w[0]
            w = padded
        return w, b

    def elaborate(self, platform):
        m = Module()

        m.submodules["lsb"] = lsb = LeftShiftBuffer(in_out_d=4)

        # build convolutions (no bus ports).
        convs = []
        for i in range(self.num_layers):
            last_layer = i == self.num_layers - 1
            w, b = self.conv_weights_biases_for(i)
            print(f"{i} CONV apply_relu={not last_layer} w {w.shape} b {b.shape}")
            # TODO: hardcoded upper bound here!
            # see https://github.com/matpalm/cached_dilated_causal_convolutions/issues/24
            conv = Conv1d(w, b, apply_relu=(not last_layer), relu_upper_bound=4.0)
            m.submodules[f"conv{i}"] = conv
            convs.append(conv)

        # register activation caches and connect the PSRAM backed ones to the
        # wishbone ports exposed by qb_network.
        activation_caches = self.activation_caches
        for i, cache in enumerate(activation_caches):
            m.submodules[f"act{i}"] = cache
        for i in self.psram_cache_indices:
            wiring.connect(
                m,
                activation_caches[i].bus,
                wiring.flipped(getattr(self, f"bus_act{i}")),
            )

        # inject cuts after each activation cache
        # to break long handshake paths.
        # TODO: do we need cuts when act cache is psram backed? very low cost i guess...
        cut_act_convs = []
        for i in range(self.num_layers - 1):  # ie. NOT last layer
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
