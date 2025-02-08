from amaranth import Elaboratable, Signal, Module
from amaranth.sim import Simulator

class LeftShiftBuffer(Elaboratable):

    def __init__(self):
        self.en = Signal()
        self.count = Signal(4)

    def elaborate(self, _platform):
        m = Module()
        with m.If(self.en):
            m.d.sync += self.count.eq(self.count + 1)
        m.d.comb += self.en.eq(self.count < 5)
        return m


# def main():
#     counter = LeftShiftBuffer()

#     async def bench(ctx):
#         while True:
#             await ctx.tick() #.repeat(5)
#             print("ctx.get(dut.count)", ctx.get(counter.count))

#     MHZ = 1e6

#     sim = Simulator(counter)
#     sim.add_clock(MHZ)
#     sim.add_testbench(bench)

#     with sim.write_vcd("output.vcd"):
#         sim.run_until(10 * MHZ)

# if __name__ == "__main__":
#     main()