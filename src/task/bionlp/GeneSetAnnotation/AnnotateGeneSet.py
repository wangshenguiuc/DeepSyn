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


if len(sys.argv) <= 2:
	pid = 1
	total_pid = 1
	dataset = 'function'
else:
	pid = int(sys.argv[1])
	dataset = str(sys.argv[2])
	total_pid = int(sys.argv[3])
	gene_set_name = int(sys.argv[4])


ImproveNet_obj,ImproveNet_obj_baseline,stop_word_list = read_ImprovedNet_obj(dataset=dataset,min_freq_cutoff=4)

Net_obj = WordNet()
go_f2f = Net_obj.ReadEdgeTypeGO()
GO_ngh_set = set()
for f in go_f2f:
	for ngh1 in go_f2f[f]:
		for ngh2 in go_f2f[f]:
			GO_ngh_set.add(ngh1+'#'+ngh2)

candidate_gene,d2g,dataset_name = read_drug_disease_to_genes(dataset=dataset)

SenGene_obj = ExtractGenSent(working_dir = repo_dir)

G2G_obj,network_gene_list = read_network_data(net_file_l = ['data/network/human/string_integrated.txt'])
ngene = len(G2G_obj.g2i)

f2g2score = {}
for f in d2g:
	g2score = np.zeros(ngene)
	ct = 0.
	for g in d2g[f]:

		if g.upper() in G2G_obj.g2i:
			gid = G2G_obj.g2i[g.upper()]
			g2score += G2G_obj.rwr[gid,:]
			ct += 1.
	if ct == 0:
		continue
	g2score /= ct
	f2g2score[f] = g2score

base_f2f = collections.defaultdict(dict)
our_f2f = collections.defaultdict(dict)
our_l = []
base_l = []

for s in d2g:
	for f in d2g:
		base_f2f[f][s] = 0.
		for g in d2g[s]:
			if f in ImproveNet_obj_baseline.net and g in ImproveNet_obj_baseline.net[f]:
				base_f2f[f][s] += ImproveNet_obj_baseline.net[f][g]['pubmed']

for max_layer in [2]:
	for edge_wt_thres in [0.08]:
		result_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'+str(max_layer)+'_'+str(edge_wt_thres)+'/' + 'result.txt'
		fin = open(result_file)
		cache =  collections.defaultdict(dict)
		for line in fin:
			f,s,r,rs,base_f2f[f][s] = line.strip().split('\t')
			cache[f][s] = str(r)+'\t'+str(rs)+'\t'+str(base_f2f[f][s])
			#fout.write(f+'\t'+s+'\t'+str(r)+'\t'+str(rs)+'\t'+str(base_f2f[f][s])+'\n')
		fin.close()
		fout = open(result_file,'w')
		for f in cache:
			for s in cache[f]:
				fout.write(f+'\t'+s+'\t'+cache[f][s]+'\n')
		for ci,ss in enumerate(d2g.keys()):
			#if ci<0.95*len(d2g.keys()):
			#	continue
			if ci%total_pid != pid and total_pid>1 :
				continue
			s = ss
			if s not in ImproveNet_obj.net or s not in f2g2score:
				#print s
				continue

			pos_gene = set(d2g[s].keys())
			exist_pos_gene = set(network_gene_list) & set(pos_gene)
			if len(exist_pos_gene) == 0:
				continue



			FN_obj = FindNeighbor(s,ImproveNet_obj,stop_word_list=stop_word_list,exclude_edge_type = [],exclude_edges = GO_ngh_set)
			g2score,gvec = FN_obj.CalNgh(G2G_obj,stop_word_list=stop_word_list,max_layer=max_layer,edge_wt_thres=edge_wt_thres,use_direct_gene_edge=True)

			if np.sum(gvec)==0:
				continue
			print ci,ci*1.0/len(d2g.keys()),ss
			#print ci,ci*1.0/len(d2g.keys())
			for i,f in enumerate(d2g.keys()):
				if f not in f2g2score:
					continue
				if f in cache and s in cache[f]:
					continue
				r = stats.spearmanr(gvec,f2g2score[f])[0]
				rs = 0.
				for g in d2g[f]:
					if g.upper() in G2G_obj.g2i:
						gid = G2G_obj.g2i[g.upper()]
						rs += gvec[gid]
				our_f2f[f][s] = r
				fout.write(f+'\t'+s+'\t'+str(r)+'\t'+str(rs)+'\t'+str(base_f2f[f][s])+'\n')
			continue

			score = []
			label = []
			baseline_score = []
			for i,f in enumerate(d2g):
				if f not in f2g2score:
					continue
				r = stats.spearmanr(gvec,f2g2score[f])[0]
				score.append(r)
				label.append(f==s)
				baseline_score.append(base_f2f[f][s])

			auc,pear,spear,auprc = evaluate_vec(score,label)
			base_auc,pear,spear,base_auprc = evaluate_vec(baseline_score,label)
			print ci,ci*1.0/len(d2g.keys()),max_layer,edge_wt_thres,s,auc,base_auc,np.mean(our_l),np.mean(base_l)
			our_l.append(auc)
			base_l.append(base_auc)
fout.close()



