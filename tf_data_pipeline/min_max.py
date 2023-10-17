import sys

MIN, MAX = None, None
for line in sys.stdin.readlines():
    line = line.strip().split(" ")
    for value in map(float, line):
        if MIN is None:
            MIN = MAX = value
        elif value < MIN:
            MIN = value
        elif value > MAX:
            MAX = value
    #print(line, MIN, MAX)

print(MIN, MAX)