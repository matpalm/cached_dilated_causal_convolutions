from pathlib import Path
import json
import tempfile

from amaranth import Module
from amaranth.lib import data, stream, wiring

from . import NNQ, NNQ_DW
from .dot_product import DotProduct


class RowByMatrixMultiply(wiring.Component):
    """Compute a matrix multiply for a single input row.

    This composes one DotProduct per output column and mirrors the
    SystemVerilog row_by_matrix_multiply wiring.
    """

    def __init__(self, weights):

        self._dot_products = [DotProduct(w) for w in weights]

        self.IN_D = self._dot_products[0].D
        self.OUT_D = len(self._dot_products)

        for dp in self._dot_products:
            if dp.D != self.IN_D:
                raise ValueError("all columns must have the same input depth")

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
