# Load some of the packages
import numpy as np

from pynats.calculator import Calculator
from pynats.data import Data
import matplotlib.pyplot as plt

import seaborn as sns

data = Data.load_dataset('forex')
calc = Calculator(dataset=data)

calc.compute()

calc.prune()

# Correlate all the features
corrmat = calc.flatten().corr(method='spearman')

# corrmat.dropna(axis=0,thres=0.2*corrmat.shape[0]).dropna(axis=1,thres=0.2*corrmat.shape[0])

sns.set(font_scale=0.8)
g = sns.clustermap(corrmat.fillna(0), mask=corrmat.isna(),
                    center=0.0,
                    cmap='RdYlBu_r',
                    xticklabels=1, yticklabels=1 )

plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.show()