# Load some of the packages
import numpy as np

from pynats.calculator import Calculator
from pynats.data import Data
import matplotlib.pyplot as plt

import seaborn as sns

data = Data(data='/home/oliver/Workspace/code/research/mvts-comparison/database/finance/SNP500/SNP500_N-367_P-11_2011-11-26-lr.npy')

# data = Data.load_dataset('forex')
calc = Calculator(dataset=data,dim_order='sp')
calc.compute()

corrmat = calc.flatten().corr(method='spearman').dropna(axis=0,how='all').dropna(axis=1,how='all')
print(f'Number of statistics left after pruning: {corrmat.shape[0]}')

sns.set(font_scale=0.5)
g = sns.clustermap( corrmat.fillna(0), mask=corrmat.isna(),
                    center=0.0,
                    cmap='RdYlBu_r',
                    xticklabels=1, yticklabels=1 )
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.show()