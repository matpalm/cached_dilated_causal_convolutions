#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--plot-png', type=str, default='plot.png')
opts = parser.parse_args()

values = sys.stdin.readlines()
values = list(map(float, values))

df = pd.DataFrame()
df['value'] = values
df['n'] = range(len(values))

sns.lineplot(df, x='n', y='value')
plt.savefig(opts.plot_png)