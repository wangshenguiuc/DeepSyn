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

dataset = 'phrase'
dataset_name = 'phrase'
ImproveNet_obj,ImproveNet_obj_baseline,stop_word_list = read_ImprovedNet_obj(dataset=dataset,min_freq_cutoff=4)

if len(sys.argv) <= 2:
	pid = 1
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])

G2G_obj,network_gene_list = read_network_data(net_file_l = ['data/network/human/string_integrated.txt'])
ngene = len(G2G_obj.g2i)


#print 'our finished'
our_f2f = collections.defaultdict(dict)

print 'base finished'
print len(network_gene_list)
for max_layer in [2,3,4,5]:
	for edge_wt_thres in [0.001,0.005,0.01,0.05,0.08,0.1]:
		result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/function_score/'+dataset_name+'/'+str(max_layer)+'_'+str(edge_wt_thres)+'/'
		#create_clean_dir(result_dir)
		if not os.path.exists(result_dir):
			os.makedirs(result_dir)
		nfunc = 0
		file = result_dir+str(pid)+'.txt'
		fout = open(file,'w')
		for ci,s in enumerate(ImproveNet_obj.net.keys()):
			if ci%total_pid != pid and total_pid>1 :
				continue
			if ImproveNet_obj.word_type[s].split('_')[1]!='function':
				continue
			nfunc+=1
			FN_obj = FindNeighbor(s,ImproveNet_obj,stop_word_list=stop_word_list,exclude_edge_type = [],exclude_edges = [],include_genes=network_gene_list)
			g2score,gvec = FN_obj.CalNgh(G2G_obj,stop_word_list=stop_word_list,max_layer=max_layer,edge_wt_thres=edge_wt_thres,all_type_same_weight=False,use_direct_gene_edge=True)

			if np.sum(gvec)==0:
				continue
			print ci,ci

			for i in range(ngene):
				g = G2G_obj.i2g[i].lower()
				fout.write(s + '\t' + g+'\t'+str(g2score[g])+'\n')
		fout.close()

