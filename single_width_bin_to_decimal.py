
from util import FxpUtil
import sys

fxp = FxpUtil()

for line in sys.stdin:
    line = '0b' + line.strip()
    binary_value = eval(line)
    val = fxp.fixed_point_to_decimal(binary_value)
    print(val)
