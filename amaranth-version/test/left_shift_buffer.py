from amaranth.sim import Simulator

from src.left_shift_buffer import LeftShiftBuffer

if __name__ == "__main__":

    counter = LeftShiftBuffer()

    async def bench(ctx):
        while True:
            await ctx.tick() #.repeat(5)
            print("ctx.get(dut.count)", ctx.get(counter.count))

    MHZ = 1e6

    sim = Simulator(counter)
    sim.add_clock(MHZ)
    sim.add_testbench(bench)

    with sim.write_vcd("output.vcd"):
        sim.run_until(10 * MHZ)
