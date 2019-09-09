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
'''
[Term]
id: GO:0000001
name: mitochondrion inheritance
namespace: biological_process
def: "The distribution of mitochondria, including the mitochondrial genome, into daughter cells after mitosis or meiosis, mediated by interactions between mitochondria and the cytoskeleton." [GOC:mcc, PMID:10873824, PMID:11389764]
synonym: "mitochondrial inheritance" EXACT []
is_a: GO:0048308 ! organelle inheritance
is_a: GO:0048311 ! mitochondrion distribution
'''
file_obo = 'data/raw_data/go.obo'
hpo_file = 'data/NLP_Dictionary/GO_term.network'
term_file = 'data/NLP_Dictionary/go_term.txt'
GO_file = 'data/function_annotation/GO.network'
fin = open(file_obo)
fout = open(hpo_file,'w')
fterm = open(term_file,'w')
fnet = open(GO_file,'w')

for line in fin:
	if line.startswith('id: '):
		GO_id = line.strip().split('id: ')[1]
	if line.startswith('name:'):
		GO = line.strip().split(': ')[1]
		fterm.write(GO+'\n')
	if line.startswith('is_a:'):
		id = line.strip().split('! ')[1]
		id1 = line.strip().split(' ! ')[0]
		id1 = id1.strip().split(': ')[1]
		fout.write(id+'\t'+GO+'\n')
		fnet.write(id1+'\t'+GO_id+'\n')
fin.close()
fout.close()
fnet.close()
fterm.close()
