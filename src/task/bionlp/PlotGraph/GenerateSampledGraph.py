import sys
import os
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'

sys.path.append('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/bionlp/PlotGraph/')
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
from plot_bionlp_figures import *
from utils import *
import operator
import time
import collections
import numpy as np
import psutil
import pcst_fast
from scipy import stats
from src.utils.evaluate.evaluate import evaluate_vec
import random


dataset = 'drug'

min_freq_cutoff=5

Net_obj = WordNet()
pubmed_word_net = {'data/Pubmed/word_network/predict_abst_181110':'infer','data/Pubmed/word_network/all_abst_181110':'pubmed'}
Net_obj.ReadWordNet(pubmed_word_net,verbal=True,min_freq_cutoff=min_freq_cutoff)

Net_obj.ReadWordType()

stop_word_file = 'data/NLP_Dictionary/stopwords.txt'
#stop_word_list = parse_word_net.GenerateStopWordList(Net_obj.word_ct,Net_obj.edge_ct,stop_word_file,Net_obj.word_type)
stop_word_list = parse_word_net.GetStopWordList(stop_word_file)
stop_word_list_manually = parse_word_net.GetStopWordList('data/NLP_Dictionary/stopwords_manually.txt')
stop_word_list = stop_word_list.union(stop_word_list_manually)
edge_list_l = [['drug','gene'],['drug','drug'],['gene','gene'],['function','gene'],['function','function'],['disease','function'],['disease','disease']]
Net_obj.ReadEdgeType(stop_word_list, edge_list_l)
selected_kg_l = [Net_obj.Monarch_d2g,Net_obj.literome_g2g,Net_obj.hpo_d2d,Net_obj.hpo_f2g,Net_obj.go_f2f,Net_obj.go_f2g]
ImproveNet_obj = ImprovedWordNet(Net_obj,selected_kg_l)

print len(ImproveNet_obj.net)

tp2ct = {}
for g in ImproveNet_obj.net:
	tp = ImproveNet_obj.word_type[g].split('_')[1]
	tp2ct[tp] = tp2ct.get(tp,0) + 1
print tp2ct

edge2ct = {}
for g in ImproveNet_obj.net:
	for g2 in ImproveNet_obj.net[g]:
		tp1 = ImproveNet_obj.word_type[g].split('_')[1]
		tp2 = ImproveNet_obj.word_type[g2].split('_')[1]
		if tp1>tp2:
			tp1,tp2=tp2,tp1
		edge2ct[tp1+'#'+tp2] = edge2ct.get(tp1+'#'+tp2,0) + 1
print edge2ct

result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/network_illustration/'
if not os.path.exists(result_dir):
	os.makedirs(result_dir)
for edge_ratio in [0.01,0.05,0.1,0.2,0.5,0.8]:
	for sample_ratio in [0.2,0.1,0.05,0.5]:
		fnode = open(result_dir + 'network_'+str(sample_ratio)+'_'+str(edge_ratio)+'_node.txt','w')
		fnode.write('source\ttype\n')
		fedge = open(result_dir + 'network_'+str(sample_ratio)+'_'+str(edge_ratio)+'_edge.txt','w')
		fedge.write('source\ttarget\tweight\ttype\n')
		node_set = set()
		for g in ImproveNet_obj.net:
			r = random.uniform(0, 1)
			if r > sample_ratio:
				continue
			node_set.add(g)
			tp = ImproveNet_obj.word_type[g].split('_')[1]
			fnode.write(g+'\t'+tp+'\n')
		for g in ImproveNet_obj.net:
			if g not in node_set:
				continue
			for g2 in ImproveNet_obj.net[g]:
				if g2 not in node_set:
					continue
				tp1 = ImproveNet_obj.word_type[g].split('_')[1]
				tp2 = ImproveNet_obj.word_type[g2].split('_')[1]
				r = random.uniform(0, 1)
				if r > edge_ratio and tp1==tp2:
					continue
				if tp1>tp2:
					tp1,tp2=tp2,tp1
				etp = tp1+'.'+tp2
				if g not in node_set or g2 not in node_set:
					print g,g2
					sys.exit(-1)
				fedge.write(g+'\t'+g2+'\t1.0\t'+etp+'\n')
		fnode.close()
		fedge.close()
