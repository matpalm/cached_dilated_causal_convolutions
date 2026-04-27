from amaranth.sim import Simulator

from left_shift_buffer import LeftShiftBuffer


def test_left_shift_buffer_matches_sv_test_sequence():
    dut = LeftShiftBuffer(w=16)

    sim = Simulator(dut)
    sim.add_clock(83e-9, domain="sync")

    async def bench(ctx):
        ctx.set(dut.rst, 0)

        last_out_0 = 0
        last_out_1 = 0
        last_out_2 = 0
        last_out_3 = 0

        # The original cocotb test samples immediately on RisingEdge, before
        # nonblocking assignments from that edge are visible.
        for i in range(10):
            ctx.set(dut.inp, i)

            last_out_0 = ctx.get(dut.out_0)
            last_out_1 = ctx.get(dut.out_1)
            last_out_2 = ctx.get(dut.out_2)
            last_out_3 = ctx.get(dut.out_3)

            await ctx.tick()

        assert last_out_0 == 5
        assert last_out_1 == 6
        assert last_out_2 == 7
        assert last_out_3 == 8

    sim.add_testbench(bench)
    sim.run()
