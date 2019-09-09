import sys
import os
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'

sys.path.append(repo_dir)
os.chdir(repo_dir)
from src.models.network_flow.PlotNetworkFlow import plot_network_flow
import cPickle as pickle
from src.datasets.FindNeighbor import FindNeighbor
from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.extractive_gen import ExtractGenSent
from src.models.generate_sentence.preprocess_word_sent import write_word_sents_to_file,get_sentence_edge,get_sentence_word,get_word_path,sentence_edge_is_in_cache
from src.models.generate_sentence.preprocess_parser_tree import preprocess_parser_tree
import operator
import time
import collections
import numpy as np
import psutil
from src.utils.evaluate.evaluate import evaluate_vec

pid = int(sys.argv[1])
total_pid = int(sys.argv[2])
min_freq_cutoff = 4

net2edge = {}
net2node = {}
net2edge['pubmed'] = set()
net2edge['infer'] = set()
net2node['pubmed'] = set()
net2node['infer'] = set()

for dataset in ['drug','disease']:
	network_dump_file = 'data/Pubmed/word_network/improved_word_net_181111_' + str(min_freq_cutoff)+'_'+dataset+'_0_0'
	ImproveNet_obj = pickle.load(open(network_dump_file, "rb" ))
	for s in ImproveNet_obj.net:
		for t in ImproveNet_obj.net[s]:
			if s<t:
				e1, e2 = s, t
			else:
				e1, e2 = t, s
			if 'pubmed' in ImproveNet_obj.net[s][t]:
				net2edge['pubmed'].add((e1,e2))
				net2node['pubmed'].add(e1)
				net2node['pubmed'].add(e2)
			if 'infer' in ImproveNet_obj.net[s][t]:
				net2edge['infer'].add((e1,e2))
				net2node['infer'].add(e1)
				net2node['infer'].add(e2)

for s in net2edge:
	print s, len(net2edge[s]),len(net2node[s])

'''
p2chunk = {}
for i in range(total_pid):
	p2chunk[i] = []
for i,c in enumerate(list(net2node['pubmed'])):
	word_path = get_word_path(c,'')[0]
	if os.path.isfile(word_path):
		continue
	p2chunk[i%total_pid].append(c)

print len(p2chunk[pid])
sys.stdout.flush()
write_word_sents_to_file(p2chunk[pid],npid=1,nchunk=400,pid=0,max_len=6)

'''


p2chunk = {}
parse_obj = preprocess_parser_tree()
for i in range(total_pid):
	p2chunk[i] = []
for i,c in enumerate(list(net2edge['pubmed'])):
	if i%10000==0:
		print i
	e1,e2 = c
	if i%total_pid == pid and sentence_edge_is_in_cache(e1,e2)[0]:
		continue
	p2chunk[i%total_pid].append(c)
print 'nedge',len(p2chunk[pid])
sys.stdout.flush()
for i,c in enumerate(list(p2chunk[pid])):
	e1,e2 = c
	sent_l = get_sentence_edge(e1,e2)
	parse_obj.get_parser_tree(e1,e2,sent_l)
	if i%1==0:
		print i*1.0/len(list(p2chunk[pid]))
