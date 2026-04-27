from amaranth.sim import Simulator

from activation_cache import ActivationCache


def _pack_two_lanes(i):
    in_0 = i & 0xFFFF
    in_1 = (i + 2) & 0xFFFF
    return (in_0 << 16) | in_1


def test_activation_cache_matches_sv_behavior():
    dut = ActivationCache(w=16, d=2, dilation=4, kernel_size=4)

    sim = Simulator(dut)
    sim.add_clock(83e-9, domain="sync")

    num_entries = dut.num_entries
    dilation = dut.dilation

    # Behavioral reference model equivalent to the original SV always block.
    ref_buffer = [0 for _ in range(num_entries)]
    ref_write_head = 0

    async def bench(ctx):
        nonlocal ref_write_head

        for i in range(120):
            inp = _pack_two_lanes(i)

            exp_l0 = ref_buffer[(ref_write_head - (3 * dilation)) % num_entries]
            exp_l1 = ref_buffer[(ref_write_head - (2 * dilation)) % num_entries]
            exp_l2 = ref_buffer[(ref_write_head - dilation) % num_entries]
            exp_l3 = inp

            ref_buffer[ref_write_head] = inp
            ref_write_head = (ref_write_head + 1) % num_entries

            ctx.set(dut.inp, inp)
            await ctx.tick()

            assert ctx.get(dut.out_l0) == exp_l0
            assert ctx.get(dut.out_l1) == exp_l1
            assert ctx.get(dut.out_l2) == exp_l2
            assert ctx.get(dut.out_l3) == exp_l3

        assert ctx.get(dut.out_l0) == 0x006B006D
        assert ctx.get(dut.out_l1) == 0x006F0071
        assert ctx.get(dut.out_l2) == 0x00730075
        assert ctx.get(dut.out_l3) == 0x00770079

    sim.add_testbench(bench)
    sim.run()
