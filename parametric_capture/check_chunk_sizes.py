import argparse
from pathlib import Path
import zarr

from common.util import zarr_base_path_for
from common.sample_db import SampleDB

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--src", type=str, required=True, nargs="+")
opts = parser.parse_args()

db = SampleDB()

for src in opts.src:
    bp = zarr_base_path_for(src)
    print(src, "|db.cv_values|", len(db.cv_values_for(src)))
    for entry in map(str, bp.iterdir()):
        if entry.endswith(".z"):  # o_O
            z = zarr.open(entry, "r")
            print(src, entry, z.nchunks)
