#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

values = sys.stdin.readlines()
values = list(map(float, values))

df = pd.DataFrame()
df['value'] = values
df['n'] = range(len(values))

sns.lineplot(df, x='n', y='value')
plt.savefig("plot.png")