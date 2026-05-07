from pathlib import Path
import sys
import unittest
import numpy as np

# for amaranth_future :/
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "amaranth_version" / "src")
)

from fxpmath_version import util as f_util
from amaranth_version.src.cdcc import util as a_util
from tf_data_pipeline.mu_law import mu_law_compress, mu_law_expand


class TestMuLaw(unittest.TestCase):

    def test_fxp_vs_amaranth_fp1_15_conversion(self):

        for val in np.arange(-0.9, 0.9, 0.01, dtype=np.float16):
            val = float(val)
            # calculate float version for fxp_math
            fxputil_1_15 = f_util.FxpUtil(n_int=1, n_frac=15)
            fxp_fp1_15 = fxputil_1_15.single_width(val)
            # calculate float version for amaranth
            ama_fp1_15 = a_util.float_to_asq(val)
            # assert same
            self.assertEqual(fxp_fp1_15, ama_fp1_15)
