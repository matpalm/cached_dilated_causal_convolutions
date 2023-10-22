#!/usr/bin/env python3

# undersample by 1/2

import sys

emit = True
for line in sys.stdin:
    if emit:
        sys.stdout.write(line)
    emit = not emit
