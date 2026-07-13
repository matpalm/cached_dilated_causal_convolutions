from numpy.typing import NDArray

from amaranth import Array, Module, Signal, signed
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

    def __init__(self, np_weights_1: NDArray, np_weights_2: NDArray | None = None):
        """
        Args:
            np_weights_1  (IN_D, OUT_D)
            np_weights_2 optional alternate bank (IN_D, OUT_D)
        """
        if len(np_weights_1.shape) != 2:
            raise Exception(
                f"Expect RowByMatrixMultiply to be inited with (OUT_D, IN_D) vector but received shape {np_weights_1.shape}"
            )
        if np_weights_2 is not None:
            if np_weights_2.shape != np_weights_1.shape:
                raise Exception(
                    "np_weights_2, if provided, needs to match shape of np_weights_1 "
                    f"{np_weights_1.shape}, but received {np_weights_2.shape}"
                )

        self.IN_D, self.OUT_D = np_weights_1.shape
        if (self.IN_D % 4 != 0) or ((self.OUT_D != 1) and (self.OUT_D % 4 != 0)):
            raise Exception(
                f"in_d={self.IN_D} and out_d={self.OUT_D} ; these must be multiples of 4; ( out_d can be 1 )"
            )

        self.num_weights = self.IN_D * self.OUT_D

        self.num_banks = 2 if np_weights_2 is not None else 1

        # TODO: is having to latch the bank here a design problem with respect to how conv1d controls it?
        self.phase = Signal(range(self.num_banks), init=0)
        self.bank_latched = Signal(range(self.num_banks), init=0)

        # Flattened as [bank][out_d][in_d], row-major within each bank.
        weight_banks = [np_weights_1]
        if np_weights_2 is not None:
            weight_banks.append(np_weights_2)

        weight_init = []
        for bank_weights in weight_banks:
            weight_rows = bank_weights.T
            for o in range(self.OUT_D):
                for i in range(self.IN_D):
                    try:
                        weight_init.append(parse_nnq(weight_rows[o][i], shape=NNQ))
                    except ValueError as e:
                        raise Exception(
                            f"!!!!!!!! weight_init o={o} i={i} {weight_rows[o][i]}", e
                        )

        self.weight_mem = Memory(
            shape=NNQ,
            depth=self.num_weights * self.num_banks,
            init=weight_init,
            attrs={"ram_style": "block"},
        )

        self.accumulator = Signal(NNQ_DW, name="rbmm_running_accum", init=0)

        self.input = Array(
            Signal(NNQ, name=f"rbmm_in_{i}", init=0) for i in range(self.IN_D)
        )
        self.i_idx = Signal(range(self.IN_D), init=0)

        self.output = Array(
            Signal(NNQ_DW, name=f"rbmm_out_{j}", init=0) for j in range(self.OUT_D)
        )
        self.o_idx = Signal(range(self.OUT_D), init=0)

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
        mul_a = Signal(signed(NNQ.width), name="rbmm_mul_a")
        mul_b = Signal(signed(NNQ.width), name="rbmm_mul_b")

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
                    # ready to process... capture input and zero output
                    for i in range(self.IN_D):
                        m.d.sync += self.input[i].eq(self.i.payload[i])
                    # TODO; we don't need to zero this (?)
                    for j in range(self.OUT_D):
                        m.d.sync += self.output[j].eq(0)
                    # reset in and out idxs and set accum=0
                    m.d.sync += [
                        self.i_idx.eq(0),
                        self.o_idx.eq(0),
                        self.accumulator.eq(0),
                        self.bank_latched.eq(self.phase),
                    ]
                    m.next = "PREFETCH_WEIGHT"

            with m.State("PREFETCH_WEIGHT"):
                # prep read of first weight ( ready for next cycle )
                m.d.comb += [
                    rd.en.eq(1),
                    rd.addr.eq(
                        self.bank_latched * self.num_weights
                        + self.o_idx * self.IN_D
                        + self.i_idx
                    ),
                ]
                m.next = "LOAD_MUL_INPUTS"

            with m.State("LOAD_MUL_INPUTS"):
                # TODO: registering mul_a and mul_b ( in a state before the actual MAC )
                #   made a huge difference to routing speed & COMB ( in a way i don't
                #   understand ). still want to revisit this re: 9x9 and ALU use
                m.d.sync += [
                    mul_a.eq(self.input[self.i_idx].as_value().as_signed()),
                    mul_b.eq(rd.data.as_value().as_signed()),
                ]
                m.next = "MAC"

            with m.State("MAC"):
                # update accumulator with latest i_idx and weight from memory
                m.d.sync += self.accumulator.eq(
                    self.accumulator.as_value().as_signed() + (mul_a * mul_b)
                )
                with m.If(self.i_idx == self.IN_D - 1):
                    # was last i_idx, write output
                    m.next = "WRITE_OUTPUT"
                with m.Else():
                    # more to do; i_idx + 1 and prep next weight read
                    m.d.sync += self.i_idx.eq(self.i_idx + 1)
                    m.d.comb += [
                        rd.en.eq(1),
                        rd.addr.eq(
                            self.bank_latched * self.num_weights
                            + self.o_idx * self.IN_D
                            + self.i_idx
                            + 1
                        ),
                    ]
                    m.next = "LOAD_MUL_INPUTS"

            with m.State("WRITE_OUTPUT"):
                # set output and reset input idx and accumulator
                m.d.sync += self.output[self.o_idx].eq(self.accumulator)
                m.d.sync += [
                    self.i_idx.eq(0),
                    self.accumulator.eq(0),
                ]
                with m.If(self.o_idx == self.OUT_D - 1):
                    # was last o_idx, we are done
                    m.next = "DONE"
                with m.Else():
                    # mode to do; o_idx + 1 and prep next weight read
                    m.d.sync += self.o_idx.eq(self.o_idx + 1)
                    m.d.comb += [
                        rd.en.eq(1),
                        rd.addr.eq(
                            self.bank_latched * self.num_weights
                            + (self.o_idx + 1) * self.IN_D
                        ),
                    ]
                    m.next = "LOAD_MUL_INPUTS"

            with m.State("DONE"):
                m.d.comb += self.o.valid.eq(1)
                with m.If(self.o.ready):
                    m.next = "IDLE"

        return m
