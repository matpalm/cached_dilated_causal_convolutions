#!/usr/bin/env python3

# resample a file from N Hz to 2N Hz
# specifically written to resample from Daisy 96kHz to FPGA 192kHz

import sys

last = None
for line in sys.stdin:
    current = [float(f) for f in line.split()]
    if last is not None:
        interpolated = [(c+l)/2 for c, l in zip(current, last)]
        print(" ".join(map(str, interpolated)))
    print(" ".join(map(str, current)))
    last = current