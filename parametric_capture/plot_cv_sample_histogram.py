# seaborn just wont shut up
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import seaborn as sns

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cv-samples-npy", type=Path, required=True)
parser.add_argument("--plot", type=Path, required=True)
opts = parser.parse_args()

cv_samples = np.load(opts.cv_samples_npy)[:, :4]
df_wide = pd.DataFrame(cv_samples, columns=["a_cv", "b_cv", "morph", "amp"])
df_long = df_wide.melt(var_name="axis", value_name="value")
g = sns.displot(
    data=df_long,
    x="value",
    col="axis",
    col_wrap=2,  # 2x2
    kde=True,
    facet_kws=dict(sharex=False, sharey=False),
)
g.savefig(opts.plot)
