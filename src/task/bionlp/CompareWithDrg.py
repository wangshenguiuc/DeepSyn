import sys
import os
from shutil import copyfile
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'


sys.path.append(repo_dir)
os.chdir(repo_dir)
dataset = 'drug'
print dataset
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


dataset = 'drug'
print dataset

if dataset == 'disease':
	dataset_name = 'Monarch_Disease'
elif dataset == 'drug':
	dataset_name = 'CTRP_GDSC_drugGene'
else:
	sys.exit('wrong dataset')
drug_cor_cutoff=0.3
fin = open('data/drug/gdsc/top_corr_genes.txt')
path2gene = {}
p2i = {}
i2p = {}
nd = 0
for line in fin:
	w = line.upper().strip().split('\t')
	cor = float(w[2])
	if abs(cor) < drug_cor_cutoff:
		continue
	d = w[1].lower()
	if d not in path2gene:
		path2gene[d] = set()
		p2i[d] = nd
		i2p[nd] = d
		nd += 1
	path2gene[d].add(w[0].lower())
fin.close()

fin = open('data/drug/ctrp/drug_map.txt')
d2dname = {}
for line in fin:
	w = line.upper().strip().split('\t')
	d2dname[w[2]] = w[0]
fin.close()
fin = open('data/NLP_Dictionary/top_genes_exp_hgnc.txt')
nd = 0
for line in fin:
	w = line.upper().strip().split('\t')
	if len(w)<3:
		continue
	cor = float(w[2])
	if abs(cor) < drug_cor_cutoff:
		continue
	d = d2dname[w[1]].lower()
	if d not in path2gene:
		path2gene[d] = set()
		p2i[d] = nd
		i2p[nd] = d
		nd += 1
	path2gene[d].add(w[0].lower())
fin.close()
print path2gene.keys()
print dataset_name
select_result_dir = result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'
result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'
for dir in os.listdir(result_dir):
	auc_our_l = []
	auc_base_l = []
	#print dir
	if not dir.startswith('4_'):
		continue
	for drug_dir in os.listdir(result_dir+dir):
		#print drug_dir
		file = result_dir + dir +'/'+ drug_dir +'/'+ drug_dir+'.txt'
		#print file
		fin = open(file)
		valid=False
		ct = 0
		gd = {}
		for line in fin:
			w = line.strip().split('\t')
			ct+=1
			if ct<=10:
				w = line.strip().split(' ')
				g = w[1]
				if drug_dir in path2gene and w[1] in path2gene[drug_dir]:
					#print w[1],drug_dir
					gd[w[1]] = line
			if w[0]=='AUROC':
				#print line
				auc_our = float(w[2])
				auc_base = float(w[3])
				if auc_our==0.5 and auc_base==0.5:
					continue
				auc_our_l.append(auc_our)
				auc_base_l.append(auc_base)
				if auc_our>0.9:
					valid =True
					#print auc_our, auc_base, file

		fin.close()
		if valid and len(gd)>0:
			print drug_dir,gd
		if valid and False:
			cur_dir = result_dir + dir +'/'+ drug_dir +'/'
			for files in os.listdir(cur_dir):
				if '.eps' in files:
					copyfile(cur_dir + files, select_result_dir+files)


	print np.mean(auc_our_l), np.mean(auc_base_l),len(auc_our_l)
