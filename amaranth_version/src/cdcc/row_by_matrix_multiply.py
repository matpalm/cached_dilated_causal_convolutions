from pathlib import Path
import json
import tempfile

from amaranth import Module
from amaranth.lib import data, stream, wiring

from . import NNQ, NNQ_DW
from .dot_product import DotProduct


class RowByMatrixMultiply(wiring.Component):

    def __init__(self, np_weights):

        if len(np_weights.shape) != 2:
            raise Exception(
                f"Expect RowByMatrixMultiply to be inited with (OUT_D, IN_D) vector but received shape {np_weights.shape}"
            )

        self._np_weights = np_weights
        self._dot_products = [DotProduct(w) for w in np_weights]

        self.OUT_D, self.IN_D = np_weights.shape

        super().__init__(
            {
                "i": wiring.In(stream.Signature(data.ArrayLayout(NNQ, self.IN_D))),
                "o": wiring.Out(stream.Signature(data.ArrayLayout(NNQ_DW, self.OUT_D))),
            }
        )

    def elaborate(self, platform):
        m = Module()

        all_cols_ready = 1
        all_cols_valid = 1

        for j, dp in enumerate(self._dot_products):
            m.submodules[f"col{j:02d}"] = dp

            m.d.comb += [
                dp.i.payload.eq(self.i.payload),
                dp.i.valid.eq(self.i.valid),
                dp.o.ready.eq(self.o.ready),
                self.o.payload[j].eq(dp.o.payload),
            ]

            all_cols_ready = all_cols_ready & dp.i.ready
            all_cols_valid = all_cols_valid & dp.o.valid

        m.d.comb += [
            self.i.ready.eq(all_cols_ready),
            self.o.valid.eq(all_cols_valid),
        ]

        return m
