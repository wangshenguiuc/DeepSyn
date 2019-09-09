import sys
import os
from shutil import copyfile
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'


sys.path.append(repo_dir)
os.chdir(repo_dir)

import cPickle as pickle
from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.extractive_gen import ExtractGenSent
from src.datasets.SubGraph import SubGraph
from src.datasets.KHopGraph import KHopGraph
import operator
import time
import collections
import numpy as np
import psutil
from src.utils.evaluate.evaluate import evaluate_vec

auc_base_l = []
auc_our_l = []


dataset = str(sys.argv[1])


if dataset == 'disease':
	dataset_name = 'Monarch_Disease'
elif dataset == 'drug':
	dataset_name = 'CTRP_GDSC_drugGene'
else:
	sys.exit('wrong dataset')


select_result_dir = result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/selected_new/'
if not os.path.exists(select_result_dir):
	os.makedirs(select_result_dir)
else:
	for subdir, dirs, files in os.walk(select_result_dir):
		for file in files:
			file_path = os.path.join(subdir, file)
			os.remove(file_path)

for topk in [2]:
	for max_layer in [4,5]:
		for edge_wt_thres in [0.01,0.03,0.05,0.08]:
			result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'+str(max_layer)+'_'+str(edge_wt_thres)+'_'+str(topk)+'/'
			for dir in os.listdir(result_dir):
				cur_file = result_dir + dir
				#print cur_file
				if not cur_file.endswith('.txt'):
					continue
				nc = 0
				fin = open(cur_file)
				w = fin.readline().strip().split('\t')
				nfunc = int(w[0])
				ngene = int(w[2])
				nnode = int(w[4])
				#g_nfunc,g_nlayer,g_ngene,g_ngene_ngh
				if nfunc<10 or ngene*1.0 / nnode < 2:
					continue
				print dir, nfunc, ngene, nnode
				if not os.path.isfile(cur_file.replace('.txt','_gset.eps')):
					continue
				copyfile(cur_file.replace('.txt','_gset.eps'), select_result_dir+str(nfunc)+'_'+str(nnode)+'_'+str(max_layer)+str(edge_wt_thres)+str(topk)+dir.replace('.txt','_gset.eps'))
