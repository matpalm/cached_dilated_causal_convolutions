import os
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import qkeras
from qkeras import quantized_bits, quantized_po2, QConv1D, QActivation

K = 4

class QKerasModelBuilder(object):

    def __init__(self, n_int: int, n_frac: int):
        self.layer_info = []

        self.n_int = n_int
        self.n_frac = n_frac

        print(f"FP N_INT={self.n_int} N_FRAC={self.n_frac}")
        self.n_word = self.n_int + self.n_frac

    # qkeras quantiser for all convolution kernels and biases
    def quantiser(self, po2: bool = False, double_width: bool = False):
        nword = self.n_word
        nint = self.n_int

        if double_width:
            nword *= 2
            nint *= 2

        if po2:
            return quantized_po2(bits=nword, max_value=1)
        else:
            return quantized_bits(bits=nword, integer=nint, alpha=1)

    # qkeras quantiser for all convolution activations
    def quant_relu(self, upper_bound: float, po2=False):
        if po2:
            assert False, "never need to use this?"
            return f"quantized_po2({self.n_word}, 1)"
        else:
            return f"quantized_relu({self.n_word},{self.n_int},relu_upper_bound={upper_bound})"

    def quant_output(self):
        return quantized_bits(bits=self.n_word, integer=self.n_int, alpha=1)

    def add_quantized_bits_conv_block(
        self,
        inp,
        layer_number: int,  # for dilation amount & naming
        out_filters: int,
        l2: float,
        relu: bool,
        relu_upper_bound: float = None,
    ):

        layer_id = f"qconv_{layer_number}_qb"
        y_pred = QConv1D(
            name=layer_id,
            filters=out_filters,
            kernel_size=K,
            padding="causal",
            dilation_rate=K**layer_number,
            kernel_quantizer=self.quantiser(),
            bias_quantizer=self.quantiser(double_width=True),
            kernel_regularizer=regularizers.L2(l2),
            bias_regularizer=regularizers.L2(l2),
        )(inp)
        self.layer_info.append({'type': 'qb', 'id': layer_id})

        if relu:
            y_pred = QActivation(
                self.quant_relu(relu_upper_bound), name=f"qrelu_{layer_number}"
            )(y_pred)
            self.layer_info.append({"type": "relu", "upper_bound": relu_upper_bound})

        return y_pred

    def add_quantized_po2_conv_block(
        self,
        inp,
        layer_number: int,  # for dilation amount & naming
        l2: float,
        out_filters: int,
        po2_filters: int,
        relu_upper_bound: float,
    ):
        # start with a _qb conv layer to handle the dilation
        layer_id = f"qconv_{layer_number}_qb"
        y_pred = QConv1D(
            name=layer_id,
            filters=out_filters,
            kernel_size=K,
            padding="causal",
            dilation_rate=K**layer_number,
            kernel_quantizer=self.quantiser(),
            bias_quantizer=self.quantiser(double_width=True),
            kernel_regularizer=regularizers.L2(l2),
            bias_regularizer=regularizers.L2(l2),
        )(inp)
        self.layer_info.append({"type": "qb", "id": layer_id})

        y_pred = QActivation(
            self.quant_relu(relu_upper_bound), name=f"qrelu_{layer_number}"
        )(y_pred)
        self.layer_info.append({"type": "relu", "upper_bound": relu_upper_bound})

        # then a pair of 1x1 _po2 convs; expand to po2_filters, contract back to out_filters
        for sublayer in [1, 2]:

            layer_id = f"qconv_{layer_number}_{sublayer}a_po2"
            y_pred = QConv1D(
                name=layer_id,
                filters=po2_filters,
                kernel_size=1,
                padding="valid",
                kernel_quantizer=self.quantiser(po2=True),
                bias_quantizer=self.quantiser(double_width=True),
            )(y_pred)
            self.layer_info.append({"type": "po2", "id": layer_id})

            layer_id = f"qconv_{layer_number}_{sublayer}b_po2"
            y_pred = QConv1D(
                name=layer_id,
                filters=out_filters,
                kernel_size=1,
                padding="valid",
                kernel_quantizer=self.quantiser(po2=True),
                bias_quantizer=self.quantiser(double_width=True),
            )(y_pred)
            self.layer_info.append({"type": "po2", "id": layer_id})

            y_pred = QActivation(
                self.quant_relu(relu_upper_bound),
                name=f"qrelu_{layer_number}_{sublayer}",
            )(y_pred)
            self.layer_info.append({"type": "relu", "upper_bound": relu_upper_bound})

        return y_pred

    def projection_layer(self, name: str, filters: int):
        return QConv1D(
            name=name,
            filters=filters,
            kernel_size=1,
            kernel_quantizer=self.quantiser(),
            bias_quantizer=self.quantiser(double_width=True),
        )

    def add_quantized_bits_regressor(
        self,
        inp,
        num_targets: int,
        # l2: float,
    ):
        y_pred = self.projection_layer("qconv_regressor_qb", num_targets)(inp)
        self.layer_info.append({"type": "regressor", "num_targets": num_targets})

        y_pred = QActivation(self.quant_output(), name="qout")(y_pred)
        self.layer_info.append(
            {"type": "qout", "n_int": self.n_int, "n_frac": self.n_frac}
        )

        return y_pred

    def create_dilated_model(
        self,
        seq_len: int,
        in_d: int,
        out_d: int,
        filter_sizes: List[int],
        # po2_filter_size: int,
        l2: float,
        relu_upper_bound: float,
        skip_project_dim: int = None,
    ):
        """
        create a qkeras model with a stack of dilation 1d convolutions

        Parameters:
            seq_len: the length of the input sequence.
            in_d: the feature dim of the input
            out_d: the feature dim of the output
            filter_sizes: output depth for each convolution layer. Number of
                layers is inferred from len(filter_sizes).
            l2: l2 penality for convolution kerne & bias
            skip_project_dim: if None, regress only on the final layer output.
                if set use this as feature dim for 1x1 projections for skips.
                note: if skip_project_dim != filter_sizes[-1] we need one extra 1x1
                to match channel widths before the residual add.
        Returns:
            qkeras model
        """

        if len(filter_sizes) == 0:
            raise ValueError("filter_sizes must contain at least one layer size")

        num_layers = len(filter_sizes)
        self.layer_info = []

        inp = Input((seq_len, in_d))
        y_pred = inp
        collected_layers = []

        for layer_num in range(num_layers):
            y_pred = self.add_quantized_bits_conv_block(
                y_pred,
                layer_number=layer_num,
                out_filters=filter_sizes[layer_num],
                l2=l2,
                relu=True,
                relu_upper_bound=relu_upper_bound,
            )
            # first layer dilates K, second K^2, etc
            self.layer_info.append(
                {
                    "type": "dilation",
                    "amount": K ** (layer_num + 1),
                    "depth": filter_sizes[layer_num],
                }
            )

            if skip_project_dim is not None:
                collected_layers.append(y_pred)

        # build the input to the regressor
        if skip_project_dim is not None:
            # each layer gets its own learned 1x1 projection into skip space;
            # all projections are summed once, then requantised once.
            skip_sum = None
            for i, layer_out in enumerate(collected_layers):
                proj = self.projection_layer(f"skip_proj_{i}", skip_project_dim)(
                    layer_out
                )
                self.layer_info.append(
                    {"type": "skip_proj", "layer": i, "dim": skip_project_dim}
                )
                if skip_sum is None:
                    skip_sum = proj
                else:
                    skip_sum = Add(name=f"skip_proj_add_{i}")([skip_sum, proj])

            skip_sum = QActivation(
                self.quant_relu(relu_upper_bound), name="skip_merge_qrelu"
            )(skip_sum)

            # if projection dim != final layer width we need one extra
            # projection so we can channel add the residual
            if skip_project_dim != filter_sizes[-1]:
                skip_sum = self.projection_layer("skip_out_proj", filter_sizes[-1])(
                    skip_sum
                )
                self.layer_info.append(
                    {"type": "skip_out_proj", "dim": filter_sizes[-1]}
                )

            # add as residual on top of the final layer output
            merged = Add(name="skip_final_add")([y_pred, skip_sum])
            merged = QActivation(
                self.quant_relu(relu_upper_bound), name="skip_final_qrelu"
            )(merged)
            regressor_inp = merged
        else:
            regressor_inp = y_pred

        y_pred = self.add_quantized_bits_regressor(regressor_inp, num_targets=out_d)

        # TODO: rewire in po2 stuff later
        # if po2_filter_size is None:
        # LKG "standard" model
        # y_pred = self.add_quantized_bits_conv_block(
        #     y_pred, layer_number=1, out_filters=filter_size, l2=l2, relu=True
        # )
        # else:
        #     # using po2
        #     y_pred = self.add_quantized_po2_conv_block(
        #         y_pred,
        #         layer_number=1,
        #         out_filters=filter_size,
        #         po2_filters=po2_filter_size,
        #         l2=l2,
        #     )
        # self.layer_info.append(
        #     {"type": "dilation", "amount": K * K, "depth": filter_size}
        # )
        # y_pred = self.add_quantized_bits_conv_block(
        #     y_pred, layer_number=2, out_filters=in_out_d, l2=l2, relu=False
        # )

        print("layer_info", self.layer_info)

        return Model(inp, y_pred)
