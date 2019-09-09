import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import collections
import operator
import pickle
import sys
import random
import os
from scipy import sparse

import cPickle as pickle
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/'
sys.path.append(repo_dir)
os.chdir(repo_dir)


from src.datasets.BioNetwork import BioNetwork
from src.datasets.FunctionAnnotation import FunctionAnnotation
import scipy.spatial as sp

file_obo = 'data/NLP_Dictionary/hp_obo_parsed.tsv'
hpo_file = 'data/NLP_Dictionary/hp_obo_format.tsv'
fin = open(file_obo)
fout = open(hpo_file,'w')
for line in fin:
	w = line.strip().lower().split('\t')
	w1 = w[0].split('(')[1].replace(')','')
	w2 = w[1].split('(')[1].replace(')','')
	fout.write(w1+'\t'+w2+'\n')
fin.close()
fout.close()
