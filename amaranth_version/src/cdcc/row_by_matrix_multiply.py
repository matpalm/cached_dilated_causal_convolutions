from numpy.typing import NDArray

from amaranth import Array, Module, Signal
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import Memory

from . import NNQ, NNQ_DW, parse_nnq

class RowByMatrixMultiply(wiring.Component):
    """Row by Matrix Multiply.

    multiples vector (IN_D) by weights of (OUT_D, IN_D)
    resulting in output (OUT_D)
    assumes IN_D and OUT_D are multiples of 4
    computes each output column sequentially.
    """

    def __init__(self, np_weights: NDArray, np_weights_alt: NDArray | None = None):
        """
        Args:
            np_weights  (IN_D, OUT_D)
            np_weights_alt optional alternate bank (IN_D, OUT_D)
        """
        if len(np_weights.shape) != 2:
            raise Exception(
                f"Expect RowByMatrixMultiply to be inited with (OUT_D, IN_D) vector but received shape {np_weights.shape}"
            )

        if np_weights_alt is not None:
            if (
                len(np_weights_alt.shape) != 2
                or np_weights_alt.shape != np_weights.shape
            ):
                raise Exception(
                    "Expect np_weights_alt to match np_weights shape "
                    f"{np_weights.shape}, but received {np_weights_alt.shape}"
                )

        # TODO: i think NUM_BANKS is overkill; changing back to just single would be good

        self.IN_D, self.OUT_D = np_weights.shape
        if (self.IN_D % 4 != 0) or (self.OUT_D % 4 != 0):
            raise Exception(
                f"in_d={self.IN_D} and out_d={self.OUT_D} ; these must be multiples of 4"
            )

        self.NUM_WEIGHTS = self.IN_D * self.OUT_D
        self.NUM_BANKS = 2 if np_weights_alt is not None else 1
        self.phase = Signal(range(self.NUM_BANKS), init=0)

        # Flattened as [bank][out_d][in_d], row-major within each bank.
        weight_banks = [np_weights]
        if np_weights_alt is not None:
            weight_banks.append(np_weights_alt)

        weight_init = []
        for bank_weights in weight_banks:
            weight_rows = bank_weights.T
            for o in range(self.OUT_D):
                for i in range(self.IN_D):
                    weight_init.append(parse_nnq(weight_rows[o][i], shape=NNQ))

        self.weight_mem = Memory(
            shape=NNQ,
            depth=self.NUM_WEIGHTS * self.NUM_BANKS,
            init=weight_init,
            attrs={"ram_style": "block"},
        )

        self.i_idx = Signal(range(self.IN_D), init=0)
        self.o_idx = Signal(range(self.OUT_D), init=0)
        self.running_accum = Signal(NNQ_DW, name="rbmm_running_accum", init=0)
        self.output = Array(
            Signal(NNQ_DW, name=f"rbmm_out_{j}", init=0) for j in range(self.OUT_D)
        )
        self.product = Signal(NNQ_DW, name="rbmm_product", init=0)
        self.bank_latched = Signal(range(self.NUM_BANKS), init=0)
        self.input = Array(
            Signal(NNQ, name=f"rbmm_in_{i}", init=0) for i in range(self.IN_D)
        )

        super().__init__(
            {
                "i": wiring.In(stream.Signature(data.ArrayLayout(NNQ, self.IN_D))),
                "o": wiring.Out(stream.Signature(data.ArrayLayout(NNQ_DW, self.OUT_D))),
            }
        )

    def elaborate(self, platform):
        m = Module()
        m.submodules["weight_mem"] = self.weight_mem

        rd = self.weight_mem.read_port(domain="sync")

        m.d.comb += [
            self.i.ready.eq(0),
            self.o.valid.eq(0),
            rd.en.eq(0),
            rd.addr.eq(0),
        ]

        for j in range(self.OUT_D):
            m.d.comb += self.o.payload[j].eq(self.output[j])

        with m.FSM():
            with m.State("IDLE"):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid & self.i.ready):
                    for i in range(self.IN_D):
                        m.d.sync += self.input[i].eq(self.i.payload[i])
                    for j in range(self.OUT_D):
                        m.d.sync += self.output[j].eq(0)
                    m.d.sync += [
                        self.i_idx.eq(0),
                        self.o_idx.eq(0),
                        self.running_accum.eq(0),
                        self.bank_latched.eq(self.phase),
                    ]
                    m.next = "PREFETCH_WEIGHT"

            with m.State("PREFETCH_WEIGHT"):
                m.d.comb += [
                    rd.en.eq(1),
                    rd.addr.eq(
                        self.bank_latched * self.NUM_WEIGHTS
                        + self.o_idx * self.IN_D
                        + self.i_idx
                    ),
                ]
                m.next = "MUL"

            with m.State("MUL"):
                m.d.sync += self.product.eq(
                    self.input[self.i_idx].as_value().as_signed()
                    * rd.data.as_value().as_signed()
                )
                m.next = "ACCUM"

            with m.State("ACCUM"):
                m.d.sync += self.running_accum.eq(
                    self.running_accum.as_value().as_signed()
                    + self.product.as_value().as_signed()
                )
                with m.If(self.i_idx == self.IN_D - 1):
                    m.next = "WRITE_OUTPUT"
                with m.Else():
                    m.d.sync += self.i_idx.eq(self.i_idx + 1)
                    m.d.comb += [
                        rd.en.eq(1),
                        rd.addr.eq(
                            self.bank_latched * self.NUM_WEIGHTS
                            + self.o_idx * self.IN_D
                            + self.i_idx
                            + 1
                        ),
                    ]
                    m.next = "MUL"

            with m.State("WRITE_OUTPUT"):
                m.d.sync += self.output[self.o_idx].eq(self.running_accum)
                m.d.sync += [
                    self.i_idx.eq(0),
                    self.running_accum.eq(0),
                ]
                with m.If(self.o_idx == self.OUT_D - 1):
                    m.next = "DONE"
                with m.Else():
                    m.d.sync += self.o_idx.eq(self.o_idx + 1)
                    m.d.comb += [
                        rd.en.eq(1),
                        rd.addr.eq(
                            self.bank_latched * self.NUM_WEIGHTS
                            + (self.o_idx + 1) * self.IN_D
                        ),
                    ]
                    m.next = "MUL"

            with m.State("DONE"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
