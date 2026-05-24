from numpy.typing import NDArray

from amaranth import Array, Cat, Const, Instance, Module, Signal, signed
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import Memory

from . import NNQ, NNQ_DW, parse_nnq


def _validate_and_build_weight_memory(np_weights, np_weights_alt):
    if len(np_weights.shape) != 2:
        raise Exception(
            f"Expect RowByMatrixMultiply to be inited with (OUT_D, IN_D) vector but received shape {np_weights.shape}"
        )

    if np_weights_alt is not None:
        if len(np_weights_alt.shape) != 2 or np_weights_alt.shape != np_weights.shape:
            raise Exception(
                "Expect np_weights_alt to match np_weights shape "
                f"{np_weights.shape}, but received {np_weights_alt.shape}"
            )

    in_d, out_d = np_weights.shape
    if (in_d % 4 != 0) or ((out_d != 1) and (out_d % 4 != 0)):
        raise Exception(
            f"in_d={in_d} and out_d={out_d} ; these must be multiples of 4; ( out_d can be 1 )"
        )

    num_weights = in_d * out_d
    num_banks = 2 if np_weights_alt is not None else 1

    # Flattened as [bank][out_d][in_d], row-major within each bank.
    weight_banks = [np_weights]
    if np_weights_alt is not None:
        weight_banks.append(np_weights_alt)

    weight_init = []
    for bank_weights in weight_banks:
        weight_rows = bank_weights.T
        for o in range(out_d):
            for i in range(in_d):
                weight_init.append(parse_nnq(weight_rows[o][i], shape=NNQ))

    weight_mem = Memory(
        shape=NNQ,
        depth=num_weights * num_banks,
        init=weight_init,
        attrs={"ram_style": "block"},
    )

    return in_d, out_d, num_weights, num_banks, weight_mem


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
        self.IN_D, self.OUT_D, self.NUM_WEIGHTS, self.NUM_BANKS, self.weight_mem = (
            _validate_and_build_weight_memory(np_weights, np_weights_alt)
        )

        self.phase = Signal(range(self.NUM_BANKS), init=0)

        self.i_idx = Signal(range(self.IN_D), init=0)
        self.o_idx = Signal(range(self.OUT_D), init=0)
        self.running_accum = Signal(NNQ_DW, name="rbmm_running_accum", init=0)
        self.output = Array(
            Signal(NNQ_DW, name=f"rbmm_out_{j}", init=0) for j in range(self.OUT_D)
        )
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
                m.next = "MAC"

            with m.State("MAC"):
                m.d.sync += self.running_accum.eq(
                    self.running_accum.as_value().as_signed()
                    + (
                        self.input[self.i_idx].as_value().as_signed()
                        * rd.data.as_value().as_signed()
                    )
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
                    m.next = "MAC"

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
                    m.next = "MAC"

            with m.State("DONE"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m


class RowByMatrixMultiplyAlu54b(wiring.Component):
    """Row by Matrix Multiply using explicit ALU54B two-lane packed path."""

    def __init__(self, np_weights: NDArray, np_weights_alt: NDArray | None = None):
        self.IN_D, self.OUT_D, self.NUM_WEIGHTS, self.NUM_BANKS, self.weight_mem = (
            _validate_and_build_weight_memory(np_weights, np_weights_alt)
        )

        self.phase = Signal(range(self.NUM_BANKS), init=0)

        self.i_idx = Signal(range(self.IN_D), init=0)
        self.o_idx = Signal(range(self.OUT_D), init=0)
        self.running_accum = Signal(NNQ_DW, name="rbmm_running_accum", init=0)
        self.output = Array(
            Signal(NNQ_DW, name=f"rbmm_out_{j}", init=0) for j in range(self.OUT_D)
        )
        self.bank_latched = Signal(range(self.NUM_BANKS), init=0)
        self.input = Array(
            Signal(NNQ, name=f"rbmm_in_{i}", init=0) for i in range(self.IN_D)
        )

        self.NNQ_W = self.input[0].shape().width
        self.NNQ_DW_W = self.running_accum.shape().width
        if self.NNQ_W > 9:
            raise Exception(
                "RowByMatrixMultiplyAlu54b requires NNQ width <= 9 for MULT9 lane packing; "
                f"received {self.NNQ_W}"
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

        rd0 = self.weight_mem.read_port(domain="sync")
        rd1 = self.weight_mem.read_port(domain="sync")

        lane_prod0 = Signal(signed(self.NNQ_DW_W), name="rbmm_lane_prod0")
        lane_prod1 = Signal(signed(self.NNQ_DW_W), name="rbmm_lane_prod1")
        mult0_p = Signal(36, name="rbmm_mult0_p")
        mult1_p = Signal(36, name="rbmm_mult1_p")
        mult0_roa = Array(Signal(name=f"rbmm_mult0_roa_{i}") for i in range(18))
        mult0_rob = Array(Signal(name=f"rbmm_mult0_rob_{i}") for i in range(18))
        mult1_roa = Array(Signal(name=f"rbmm_mult1_roa_{i}") for i in range(18))
        mult1_rob = Array(Signal(name=f"rbmm_mult1_rob_{i}") for i in range(18))
        mult0_signedp = Signal(name="rbmm_mult0_signedp")
        mult1_signedp = Signal(name="rbmm_mult1_signedp")
        alu_r = Signal(54, name="rbmm_alu_r")
        a0_ext = Signal(signed(18), name="rbmm_a0_ext")
        a1_ext = Signal(signed(18), name="rbmm_a1_ext")
        b0_ext = Signal(signed(18), name="rbmm_b0_ext")
        b1_ext = Signal(signed(18), name="rbmm_b1_ext")

        for j in range(self.OUT_D):
            m.d.comb += self.o.payload[j].eq(self.output[j])

        m.d.comb += [
            self.i.ready.eq(0),
            self.o.valid.eq(0),
            rd0.en.eq(0),
            rd0.addr.eq(0),
            rd1.en.eq(0),
            rd1.addr.eq(0),
            a0_ext.eq(self.input[self.i_idx].as_value().as_signed()),
            a1_ext.eq(self.input[self.i_idx + 1].as_value().as_signed()),
            b0_ext.eq(rd0.data.as_value().as_signed()),
            b1_ext.eq(rd1.data.as_value().as_signed()),
            lane_prod0.eq(mult0_p[: self.NNQ_DW_W].as_signed()),
            lane_prod1.eq(mult1_p[: self.NNQ_DW_W].as_signed()),
        ]

        mult0_kwargs = {
            "i_SIGNEDA": Const(1),
            "i_SIGNEDB": Const(1),
            "i_SOURCEA": Const(0),
            "i_SOURCEB": Const(0),
        }
        mult1_kwargs = {
            "i_SIGNEDA": Const(1),
            "i_SIGNEDB": Const(1),
            "i_SOURCEA": Const(0),
            "i_SOURCEB": Const(0),
        }
        for i in range(4):
            mult0_kwargs[f"i_CLK{i}"] = Const(0)
            mult0_kwargs[f"i_CE{i}"] = Const(0)
            mult0_kwargs[f"i_RST{i}"] = Const(0)
            mult1_kwargs[f"i_CLK{i}"] = Const(0)
            mult1_kwargs[f"i_CE{i}"] = Const(0)
            mult1_kwargs[f"i_RST{i}"] = Const(0)
        for i in range(18):
            mult0_kwargs[f"i_C{i}"] = Const(0)
            mult0_kwargs[f"i_SRIA{i}"] = Const(0)
            mult0_kwargs[f"i_SRIB{i}"] = Const(0)
            mult0_kwargs[f"o_ROA{i}"] = mult0_roa[i]
            mult0_kwargs[f"o_ROB{i}"] = mult0_rob[i]
            mult0_kwargs[f"o_ROC{i}"] = Signal(name=f"rbmm_mult0_roc_{i}")
            mult0_kwargs[f"o_SROA{i}"] = Signal(name=f"rbmm_mult0_sroa_{i}")
            mult0_kwargs[f"o_SROB{i}"] = Signal(name=f"rbmm_mult0_srob_{i}")

            mult1_kwargs[f"i_C{i}"] = Const(0)
            mult1_kwargs[f"i_SRIA{i}"] = Const(0)
            mult1_kwargs[f"i_SRIB{i}"] = Const(0)
            mult1_kwargs[f"o_ROA{i}"] = mult1_roa[i]
            mult1_kwargs[f"o_ROB{i}"] = mult1_rob[i]
            mult1_kwargs[f"o_ROC{i}"] = Signal(name=f"rbmm_mult1_roc_{i}")
            mult1_kwargs[f"o_SROA{i}"] = Signal(name=f"rbmm_mult1_sroa_{i}")
            mult1_kwargs[f"o_SROB{i}"] = Signal(name=f"rbmm_mult1_srob_{i}")
        for i in range(18):
            mult0_kwargs[f"i_A{i}"] = a0_ext[i]
            mult0_kwargs[f"i_B{i}"] = b0_ext[i]
            mult1_kwargs[f"i_A{i}"] = a1_ext[i]
            mult1_kwargs[f"i_B{i}"] = b1_ext[i]
        for i in range(36):
            mult0_kwargs[f"o_P{i}"] = mult0_p[i]
            mult1_kwargs[f"o_P{i}"] = mult1_p[i]
        mult0_kwargs["o_SIGNEDP"] = mult0_signedp
        mult1_kwargs["o_SIGNEDP"] = mult1_signedp
        m.submodules["rbmm_mult0"] = Instance("MULT18X18D", **mult0_kwargs)
        m.submodules["rbmm_mult1"] = Instance("MULT18X18D", **mult1_kwargs)

        kwargs = {
            "a_keep": 1,
            "p_MULT9_MODE": "ENABLED",
            "p_LEGACY": "DISABLED",
            "i_SIGNEDIA": mult0_signedp,
            "i_SIGNEDIB": mult1_signedp,
            "i_SIGNEDCIN": Const(0),
        }
        for i in range(4):
            kwargs[f"i_CLK{i}"] = Const(0)
            kwargs[f"i_CE{i}"] = Const(0)
            kwargs[f"i_RST{i}"] = Const(0)
        for i in range(11):
            kwargs[f"i_OP{i}"] = Const(0)
        for i in range(18):
            kwargs[f"i_A{i}"] = mult0_roa[i]
            kwargs[f"i_A{i + 18}"] = mult0_rob[i]
            kwargs[f"i_B{i}"] = mult1_roa[i]
            kwargs[f"i_B{i + 18}"] = mult1_rob[i]
        for i in range(36):
            kwargs[f"i_MA{i}"] = mult0_p[i]
            kwargs[f"i_MB{i}"] = mult1_p[i]
        for i in range(54):
            kwargs[f"i_C{i}"] = Const(0)
            kwargs[f"i_CFB{i}"] = Const(0)
            kwargs[f"i_CIN{i}"] = Const(0)
            kwargs[f"o_R{i}"] = alu_r[i]
        m.submodules["rbmm_alu54b"] = Instance("ALU54B", **kwargs)

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
                    rd0.en.eq(1),
                    rd0.addr.eq(
                        self.bank_latched * self.NUM_WEIGHTS
                        + self.o_idx * self.IN_D
                        + self.i_idx
                    ),
                    rd1.en.eq(1),
                    rd1.addr.eq(
                        self.bank_latched * self.NUM_WEIGHTS
                        + self.o_idx * self.IN_D
                        + self.i_idx
                        + 1
                    ),
                ]
                m.next = "MAC"

            with m.State("MAC"):
                m.d.sync += self.running_accum.eq(
                    self.running_accum.as_value().as_signed() + lane_prod0 + lane_prod1
                )
                with m.If(self.i_idx == self.IN_D - 2):
                    m.next = "WRITE_OUTPUT"
                with m.Else():
                    m.d.sync += self.i_idx.eq(self.i_idx + 2)
                    m.d.comb += [
                        rd0.en.eq(1),
                        rd0.addr.eq(
                            self.bank_latched * self.NUM_WEIGHTS
                            + self.o_idx * self.IN_D
                            + self.i_idx
                            + 2
                        ),
                        rd1.en.eq(1),
                        rd1.addr.eq(
                            self.bank_latched * self.NUM_WEIGHTS
                            + self.o_idx * self.IN_D
                            + self.i_idx
                            + 3
                        ),
                    ]
                    m.next = "MAC"

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
                        rd0.en.eq(1),
                        rd0.addr.eq(
                            self.bank_latched * self.NUM_WEIGHTS
                            + (self.o_idx + 1) * self.IN_D
                        ),
                        rd1.en.eq(1),
                        rd1.addr.eq(
                            self.bank_latched * self.NUM_WEIGHTS
                            + (self.o_idx + 1) * self.IN_D
                            + 1
                        ),
                    ]
                    m.next = "MAC"

            with m.State("DONE"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
