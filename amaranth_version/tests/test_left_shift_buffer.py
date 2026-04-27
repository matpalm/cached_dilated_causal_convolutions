from pathlib import Path
import sys

from amaranth.sim import Simulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cdcc.left_shift_buffer import LeftShiftBuffer


def test_left_shift_buffer():
    dut = LeftShiftBuffer()

    ref = [[0 for _ in range(dut.OUT_D)] for _ in range(dut.K)]

    async def testbench(ctx):
        ctx.set(dut.o.ready, 1)

        for i in range(10):
            in_row = [i % 4, (i + 1) % 4, (i + 2) % 4, 0]

            ctx.set(dut.i.valid, 1)
            for d in range(dut.IN_D):
                ctx.set(dut.i.payload[d], in_row[d])

            assert ctx.get(dut.i.ready) == 1
            assert ctx.get(dut.o.valid) == 1

            await ctx.tick()

            ref.pop(0)
            ref.append(in_row)

            for k in range(dut.K):
                for d in range(dut.OUT_D):
                    assert ctx.get(dut.o.payload[k][d]).as_float() == ref[k][d]

        ctx.set(dut.i.valid, 1)
        for d in range(dut.IN_D):
            ctx.set(dut.i.payload[d], 3)

        ctx.set(dut.o.ready, 0)
        await ctx.tick()

        assert ctx.get(dut.i.ready) == 0
        assert ctx.get(dut.o.valid) == 1

        for k in range(dut.K):
            for d in range(dut.OUT_D):
                assert ctx.get(dut.o.payload[k][d]).as_float() == ref[k][d]

    sim = Simulator(dut)
    sim.add_clock(1e-6, domain="sync")
    sim.add_testbench(testbench)
    with sim.write_vcd(vcd_file=open("test_lsb.vcd", "w")):
        sim.run()
