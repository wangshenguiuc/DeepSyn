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

npid = 400
G2G_obj,network_gene_list = read_network_data(net_file_l = ['data/network/human/string_integrated.txt'])
ngene = len(G2G_obj.g2i)

fout = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/function_score/phrase/2_0.01/all_new.txt','w')
for i in range(npid):
	print i
	s2g = collections.defaultdict(dict)
	file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/function_score/phrase/2_0.01/'+str(i)+'.txt'
	if not os.path.isfile(file):
		print file
		continue
	fin = open(file)
	for line in fin:
		s,g,sc = line.strip().split('\t')
		s2g[s][g] = sc
	fin.close()
	for s in s2g:
		fout.write(s)
		for i in range(ngene):
			g = G2G_obj.i2g[i].lower()
			if g not in s2g[s]:
				print i,g
			fout.write('\t'+str(s2g[s][g]))
		fout.write('\n')
		#fout.write(s + '\t' + g+'\t'+str(g2score[g])+'\n')
fout.close()
