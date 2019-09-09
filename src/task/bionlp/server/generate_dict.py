import sys
import os
from configure import hyperpara
repo_dir = hyperpara['repo_dir']
sys.path.append(repo_dir + 'src/task/bionlp/')
sys.path.append(repo_dir)
os.chdir(repo_dir)
import cPickle as pickle
from src.datasets.BioNetwork import BioNetwork
from src.datasets.FindLayer import FindLayer
from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.extractive_gen import ExtractGenSent
from utils import *
import operator
import time
import collections
import numpy as np
import psutil
import networkx as nx
import pcst_fast
from src.utils.evaluate.evaluate import evaluate_vec
from src.models.network_flow.PlotNetworkFlow import plot_network_flow
#n/9EaiRm
DATA_DIR = '/data/cellardata/users/netant/Data/'
CACHE_DIR = '/data/cellardata/users/netant/Cache/'
'''
*, don't care, specific genes
p-value, care or not care cateogry

disease -> function -> gene set -> drug
1. disease, rank gene.
2. drug, rank gene.
3. gene set -> GO
4. GWAS -> GO
5. GO, rank gene
hyperpara = {
#'repo_dir':'/oak/stanford/groups/rbaltman/swang91/Sheng_repo/',
'repo_dir':'/cellar/users/majianzhu/Data/wangsheng/NetAnt/',
'max_layer':4,
'edge_wt_thres':0.01,
'net_topk':3,
'max_end_nodes':100,
'min_freq_cutoff':15
}

'''


#python run_server.py #cisplatin# #abnormality of the glial cells#glimo# #p21# #cell cycle#g1 phase#

def load_data(DATA_DIR,CACHE_DIR, dataset='disease',min_freq_cutoff=100):
	''''
	dataset: query a drug or a disease

	return:
	G2G_obj: Gene network object
	network_gene_list: list of genes in the network
	KnowledgeGraph_obj: knowledge graph
	stop_word_list: stop words for NLP
	'''
	KnowledgeGraph_obj,KnowledgeGraph_obj_baseline,stop_word_list = read_ImprovedNet_obj(dataset=dataset,min_freq_cutoff=min_freq_cutoff,read_baseline=False,DATA_DIR=DATA_DIR,CACHE_DIR=CACHE_DIR)
	Net_obj = WordNet()
	#SenGene_obj = ExtractGenSent(working_dir = repo_dir)
	G2G_obj,network_gene_list = read_network_data(DATA_DIR=DATA_DIR,net_file_l = [DATA_DIR + 'data/network/human/string_integrated.txt'])

	return network_gene_list,G2G_obj,KnowledgeGraph_obj,stop_word_list


min_freq_cutoff = 15
network_gene_list,G2G_obj,KnowledgeGraph_obj,stop_word_list=load_data(DATA_DIR,CACHE_DIR, dataset = 'disease',min_freq_cutoff=min_freq_cutoff)
fout = open(CACHE_DIR+'all_words_type_freq_gt_'+str(min_freq_cutoff)+'.txt','w')
for w in KnowledgeGraph_obj.word_type:
	source,tp = KnowledgeGraph_obj.word_type[w].split('_')
	fout.write(w+'\t'+tp+'\t'+source+'\n')
fout.close()
