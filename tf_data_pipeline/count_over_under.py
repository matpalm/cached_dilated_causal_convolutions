import sys

n_other = 0
n_under = 0
n_over = 0

THRESHOLD = 0.70

for line in sys.stdin.readlines():
    line = line.strip().split(" ")
    for value in map(float, line):
        if value < -THRESHOLD:
            n_under += 1
        elif value > THRESHOLD:
            n_over += 1
        else:
            n_other += 1

print("n_other", n_other, "n_under", n_under, "n_over", n_over)