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


dataset = 'everything'

min_freq_cutoff=5
network_dump_file = 'data/Pubmed/word_network/improved_word_net_181112_' + str(min_freq_cutoff)+'_'+dataset
if os.path.isfile(network_dump_file):
	ImproveNet_obj = pickle.load(open(network_dump_file, "rb" ))
else:
	Net_obj = WordNet()
	pubmed_word_net = {'data/Pubmed/word_network/predict_abst_181110':'infer','data/Pubmed/word_network/all_abst_181110':'pubmed'}
	Net_obj.ReadWordNet(pubmed_word_net,verbal=True,min_freq_cutoff=min_freq_cutoff)

	Net_obj.ReadWordType()

	stop_word_file = 'data/NLP_Dictionary/stopwords.txt'
	#stop_word_list = parse_word_net.GenerateStopWordList(Net_obj.word_ct,Net_obj.edge_ct,stop_word_file,Net_obj.word_type)
	stop_word_list = parse_word_net.GetStopWordList(stop_word_file)
	stop_word_list_manually = parse_word_net.GetStopWordList('data/NLP_Dictionary/stopwords_manually.txt')
	stop_word_list = stop_word_list.union(stop_word_list_manually)
	edge_list_l = [['drug','disease'],['drug','tisue'],['drug','gene'],['drug','function'],['tisue','function'],['tisue','disease'],['disease','function'],['disease','tisue'],['disease','disease'],['function','function'],['gene','gene'],
				['function','gene'],['tisue','gene'],['disease','gene']]
	Net_obj.ReadEdgeType(stop_word_list, edge_list_l)
	selected_kg_l = [Net_obj.Monarch_d2g,Net_obj.literome_g2g,Net_obj.hpo_d2d,Net_obj.hpo_f2g,Net_obj.go_f2f,Net_obj.go_f2g]
	ImproveNet_obj = ImprovedWordNet(Net_obj,selected_kg_l)
	ImproveNet_obj.reload()
	with open(network_dump_file, 'wb') as output:
	   pickle.dump(ImproveNet_obj, output, pickle.HIGHEST_PROTOCOL)

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

result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/selected_subnetwork_illustration/'
if not os.path.exists(result_dir):
	os.makedirs(result_dir)
valid = 0
for g in ImproveNet_obj.net:
	tp = ImproveNet_obj.word_type[g].split('_')[1]
	if tp!='drug' and tp!='disease':
		continue
	tp2ct = {}
	tp2ct[tp] = 1
	node_set = set()
	node_set.add(g)
	for ngh in ImproveNet_obj.net[g]:
		tp = ImproveNet_obj.word_type[ngh].split('_')[1]
		tp2ct[tp] = tp2ct.get(tp,0)+1
		node_set.add(ngh)
	edge_ct = 0.
	nnode = len(node_set)
	edge_type = {}
	edge_list = []
	for n1 in node_set:
		for n2 in node_set:
			if n2 in ImproveNet_obj.net and n1 in ImproveNet_obj.net[n2]:
				edge_ct+=1
				for tp in ImproveNet_obj.net[n2][n1]:
					edge_type[tp] = edge_type.get(tp,0)+1
					if [n1,n2,tp] not in edge_list:
						edge_list.append([n2,n1,tp])
	ratio = edge_ct / nnode / nnode
	if len(node_set)<18 or len(node_set)>25 or len(edge_type)<3 or np.min(edge_type.values())<=3 or len(tp2ct)<4:
		continue
	print valid,ratio,g,edge_type,tp2ct
	valid+=1
	#continue
	fnode = open(result_dir + g+'_node.txt','w')
	fnode.write('source\ttype\n')
	fedge = open(result_dir + g+'_edge.txt','w')
	fedge.write('source\ttarget\tweight\ttype\n')
	for g in node_set:
		tp = ImproveNet_obj.word_type[g].split('_')[1]
		fnode.write(g+'\t'+tp+'\n')
	for e in edge_list:
		g1,g2,tp = e
		fedge.write(g1+'\t'+g2+'\t'+str(1.0)+'\t'+tp+'\n')
	fnode.close()
	fedge.close()
