from pathlib import Path
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--top-tim", type=Path, required=True)
opts = parser.parse_args()
print("opts", opts)

state = "START"


def interesting_du_line(line):
    for s in ["DP16KD", "MULT18X18D", "ALU54B", "TRELLIS_FF", "TRELLIS_COMB"]:
        if s in line:
            return True
    return False


last_line = None
with open(opts.top_tim, "r") as f:
    for line in f.readlines():
        line = line.strip()

        if line.startswith("Info: Max frequency for clock"):
            print(line)
        elif line.startswith("Info: Router1 time"):
            print(line)
        else:
            match state:
                case "START":
                    if line == "Info: Device utilisation:":
                        state = "DEVICE_UTIL"
                case "DEVICE_UTIL":
                    if len(line) == 0:
                        state = "TIMING"
                    elif interesting_du_line(line):
                        print("DU", line)
