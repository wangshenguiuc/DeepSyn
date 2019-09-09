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
#warnings.simplefilter(action='ignore', category=FutureWarning)
net_file_l = []
net_file_l.append(data_dir + 'network/human/string_integrated.txt')
Net_obj = BioNetwork(net_file_l)
network = Net_obj.sparse_network.toarray()
i2g = Net_obj.i2g
g2i = Net_obj.g2i
nnode = len(i2g)

GO_file_l = [data_dir + 'function_annotation/GO.network']
GO_obj = BioNetwork(GO_file_l,reverse=True)
GO_net = GO_obj.network_d[GO_file_l[0]]
GO_rev_obj = BioNetwork(GO_file_l,reverse=False)
GO_net_rev = GO_rev_obj.network_d[GO_file_l[0]]
fin = open(data_dir+'function_annotation/GO2name.txt')
GO2name  ={}
name2GO  ={}
for line in fin:
    w  = line.strip().split('\t')
    if len(w) < 2:
        continue
    GO2name[w[0]] = w[1]
    name2GO[w[1]] = w[0]
fin.close()
Func_obj = FunctionAnnotation(data_dir + 'function_annotation/gene_association.goa_human', GO_net)
#nfunc = len(Func_obj.f2g)
path2gene = Func_obj.f2g
p2i = Func_obj.f2i
ct=0.
fout = open(data_dir + 'NLP_Dictionary/go2gene_set_complete.txt','w')
for f in path2gene:
	#print f
	if f not in GO2name:
		print f,ct
		ct+=1
		continue
	fout.write(GO2name[f])
	for g in path2gene[f]:
		fout.write('\t'+g)
	fout.write('\n')
fout.close()
'''
