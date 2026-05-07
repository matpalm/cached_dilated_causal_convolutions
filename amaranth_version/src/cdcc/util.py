from amaranth_future import fixed
import os

_ASQ_WIDTH = int(os.environ.get("TILIQUA_ASQ_WIDTH", "16"))
_ASQ_I_BITS = int(os.environ.get("TILIQUA_ASQ_I_BITS", "1"))
ASQ = fixed.SQ(_ASQ_I_BITS, _ASQ_WIDTH - _ASQ_I_BITS)


def float_to_asq(v: float):
    return fixed.Const(v, ASQ, clamp=True).as_float()
