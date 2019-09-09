import sys
import os
import pandas as PD
import numpy as np
import cPickle as pickle
from sklearn.linear_model import Ridge,ElasticNet
from scipy import stats
import collections
from sklearn.model_selection import GridSearchCV
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'

sys.path.append(repo_dir)
os.chdir(repo_dir)
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'
n_groups = 2
means_frank = (0.92,0.819)
means_guido = (0.57,0.596)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4

plt.style.use('ggplot')

rects1 = plt.bar(index, means_frank, bar_width,
                 alpha=opacity,color='b',
                 label='Our method')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                 alpha=opacity,color='r',
                 label='Baseline')

plt.xlabel('Dataset')
plt.ylabel('AUROC')
plt.ylim([0.5,1.0])
plt.title('Comparison of our method with the simple NLP approach')
plt.xticks(index + bar_width, ('Drug Target (n=111)', 'Disease genes (n=1441)'))
plt.legend()

plt.tight_layout()

plt.savefig(result_dir+'comparison_our_method.pdf')
