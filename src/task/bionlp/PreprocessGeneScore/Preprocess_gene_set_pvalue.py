import sys
import os
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
sys.path.append('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/bionlp/')
sys.path.append(repo_dir)
os.chdir(repo_dir)
from src.models.network_flow.PlotNetworkFlow import plot_network_flow
import cPickle as pickle
from src.datasets.BioNetwork import BioNetwork
from src.datasets.FindNeighbor import FindNeighbor
from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.extractive_gen import ExtractGenSent
from src.models.pathway_enrichment.PathwayDrugResponse import PathwayDrugResponse
import operator
import time
import collections
import numpy as np
import psutil
import pcst_fast
from scipy import stats
from src.utils.evaluate.evaluate import evaluate_vec
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from src.models.pathway_enrichment.PathwayEnrichment import PathwayEnrichment
from utils import *

gs = []

result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/function_score/pvalue/'
if not os.path.exists(result_dir):
	os.makedirs(result_dir)

fin = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/function_score/phrase/2_0.01/all_new.txt')
for line in fin:
	w = line.strip().split('\t')
	for i in range(1,len(w)):
		if float(w[i])!=0:
			gs.append(float(w[i]))
	if len(gs) > 1e8:
		break
fin.close()
print len(gs)
nsample = 100000
gs = np.array(gs)
for nsample in [1e5,1e6,1e7,1e8]:
	fout = open(result_dir+str(int(nsample)),'w')
	s_l = np.random.choice(gs, nsample,replace =True)
	s_l = np.sort(np.array(s_l))
	print nsample
	for s in s_l:
		fout.write(str(s)+'\n')
	fout.close()
	print nsample
