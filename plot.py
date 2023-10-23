#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import warnings

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

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    p = sns.lineplot(df, x='n', y='value')
    p.set(ylim=(-2, 2))
    plt.savefig(opts.plot_png)