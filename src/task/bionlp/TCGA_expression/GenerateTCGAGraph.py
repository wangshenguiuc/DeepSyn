import sys
import os
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
sys.path.append('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/bionlp/')
sys.path.append(repo_dir)
os.chdir(repo_dir)
from src.models.network_flow.PlotNetworkFlow import plot_network_flow,write_network_cytoscape
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
import seaborn as sns
from adjustText import adjust_text
import seaborn as sns
from scipy import stats
from src.utils.evaluate.evaluate import evaluate_vec
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from src.models.pathway_enrichment.PathwayEnrichment import PathwayEnrichment
from utils import *


def augmented_dendrogram(linkage_matrix,gene_set_l,color_threshold=1,p=2,truncate_mode='lastp',show_leaf_counts=True):
	ddata = dendrogram(linkage_matrix,color_threshold=color_threshold,p=p,truncate_mode=truncate_mode,show_leaf_counts=show_leaf_counts)
	y_l = []
	for y in ddata['dcoord']:
		y_l.append(y[1])
	y_l = np.array(y_l)
	y_ind = np.argsort(y_l)
	#new_gene_set_l = []
	#for k in y_ind:
	#	new_gene_set_l.append(gene_set_l[k])

	k=0
	texts = []
	for i, d in zip(ddata['icoord'], ddata['dcoord']):
		x = 0.5 * sum(i[1:3])
		y = d[1]
		plt.plot(x, y, 'ro')
		#if new_gene_set_l[k].startswith('@'):
		#	c= 'red'
		#else:
		#	c= 'blue'
		#new_gene_set_l[k]
		#texts.append(plt.text( x, y, new_gene_set_l[k], ha='center', wrap=True,color=c,fontsize=5))
		k+=1
	#adjust_text(texts,  only_move={'points':'y', 'text':'y', 'objects':'y'})#, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
	return ddata

def query(gene_set,candidate_term,f2g_mat,G2G_obj,background_sc,kword=5,stop_word = []):
	term_set = []
	sc = {}
	for f in candidate_term:
		if f in stop_word:
			continue
		if f not in f2g_mat:
			continue
		sc[f] = 0.
		ngene_term = 0
		for g in gene_set:
			if g in f2g_mat[f]:
				sc[f]+= f2g_mat[f][g]
				ngene_term+=1
		if ngene_term==0:
			continue
		sc[f] /= ngene_term
	sorted_x = sorted(sc.items(), key=operator.itemgetter(1))
	sorted_x.reverse()
	sign_term_set = {}
	best_term = 'NA'
	best_pv = 2
	for x in sorted_x:
		f = x[0]
		score = sc[f]
		if f in sc:
			pv = (len(np.where(background_sc>score)[0]) + 1) * 1. / len(background_sc)

			if pv > 0.05:
				break
		else:
			pv = 1
		if pv < best_pv:
			best_pv = pv
			best_term = f
		if pv < 0.05:
			sign_term_set[f] = pv
	return sign_term_set

def read_all_func_data(gene_set,G2G_obj):
	f2g_sc =  collections.defaultdict(dict)
	fin = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/function_score/phrase/2_0.01/all.txt')
	for line in fin:
		w = line.strip().split('\t')
		d = w[0]
		for g in gene_set:
			if g not in G2G_obj.g2i:
				continue
			i = G2G_obj.g2i[g]
			f2g_sc[d][g] = float(w[i+1])
	fin.close()
	return f2g_sc


if len(sys.argv) <= 2:
	pid = 1
	total_pid = 1
else:
	pid = int(sys.argv[1])
	total_pid = int(sys.argv[2])
'''
pvalue_dec = 1e6
result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/function_score/pvalue/'
fin = open(result_dir+str(int(pvalue_dec)))
background_sc = []
for line in fin:
	w = line.strip()
	background_sc.append(float(w))
fin.close()
background_sc = np.array(background_sc)

dataset = 'everything'
ImproveNet_obj,ImproveNet_obj_baseline,stop_word_list = read_ImprovedNet_obj(dataset=dataset,min_freq_cutoff=4,max_ngh=100)

#stop_word = []
stop_word_list.add('c6 glioma cell')
stop_word_list.add('stem cell phenotype')
stop_word_list.add('gbm tumors')
stop_word_list.add('glioma cell growth')
stop_word_list.add('primary brain tumor')
stop_word_list.add('vegf family')
stop_word_list.add('organ growth')
stop_word_list.add('neuroepithelial tumors')
stop_word_list.add('glial progenitor cells')
stop_word_list.add('pancreatic cancer tissues')
stop_word_list.add('histone methylation')
stop_word_list.add('human glioblastoma cell line')
stop_word_list.add('cd95 fas')
stop_word_list.add('inhibiting angiogenesis')
stop_word_list.add('her2 her3')
stop_word_list.add('gli1 expression')
stop_word_list.add('expressed nestin')
stop_word_list.add('neurosphere formation')
stop_word_list.add('mouse astrocytes')
stop_word_list.add('inclusion formation')
stop_word_list.add('glioblastoma cell line')
stop_word_list.add('human glioblastoma cell lines')
stop_word_list.add('pten mutations')
stop_word_list.add('idh1 mutation')
stop_word_list.add('glia1 tumor')
'''
G2G_obj,network_gene_list = read_network_data(net_file_l = ['data/network/human/string_integrated.txt'])
ngene = len(G2G_obj.g2i)




fin = open(repo_dir+'data/TCGA/cancer2name.txt')
c2n = {}
for line in fin:
	c,n = line.strip().split('\t')
	c2n[n] = c
fin.close()
our_f2f = collections.defaultdict(dict)

max_layer = 4
edge_wt_thres = 0.01

topk = 100
max_clst = 20
expression_folder = 'data/TCGA/expression/'
exp_file_l = os.listdir(expression_folder)

term_id = 83

include_terms = set()
include_terms.add('proteasomal inhibition')
include_terms.add('cell autophagy')
include_terms.add('vessel maturation')
include_terms.add('astrocyte differentiation')
#include_terms.add('histone deacetylation')
include_terms.add('catabolic process')
#include_terms.add('methyl-cpg binding')
#include_terms.add('apoptosome')
include_terms.add('glial cell migration')
include_terms.add('nuclear body')
for ci,exp_file in enumerate(exp_file_l):
	if ci%total_pid != pid and total_pid>1 :
		continue
	cancer = exp_file.replace('.txt','')
	if cancer not in c2n:
		continue
	cname = c2n[cancer]
	if cancer!='GBM':
		continue
	plot_result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/TCGA_expression/' + cancer + '/'
	if not os.path.exists(plot_result_dir):
		os.makedirs(plot_result_dir)
	p2g_exp = collections.defaultdict(dict)
	fin = open('data/TCGA/expression/' + exp_file)
	p2i = {}
	g2i = {}
	i2p = {}
	i2g = {}
	for line in fin:
		p,g,v = line.upper().strip().split('\t')
		if g not in G2G_obj.g2i:
			continue
		p2g_exp[p][g] = np.log(float(v)+1)
		if p not in p2i:
			i = len(p2i)
			p2i[p] = i
			i2p[i] = p
		if g not in g2i:
			i = len(g2i)
			i2g[i] = g
			g2i[g] = i
	fin.close()
	npat = len(p2i)
	ngene = len(g2i)

	p2g_mat = np.zeros((npat,ngene))
	for p in p2g_exp:
		for g in p2g_exp[p]:
			p2g_mat[p2i[p], g2i[g]] = p2g_exp[p][g]
	gstd = np.std(p2g_mat,axis=0)
	ind = np.argsort(gstd*-1)[:min(topk,ngene)]

	tick = []
	select_gene_set = set()
	for id in ind:
		select_gene_set.add(i2g[id])
		tick.append(i2g[id])

	PE = PathwayEnrichment(pathfile_l=['data/pathway/gene_ontology.txt'],gene_set = set(g2i.keys()))

	linkage_matrix = linkage(p2g_mat[:,ind].T, "ward")

	plt.clf()
	row_linkage = linkage(p2g_mat[:,ind], "ward")
	col_linkage = linkage(p2g_mat[:,ind].T, "ward")
	sns.set(font_scale=0.5)
	sns_plot = sns.clustermap(p2g_mat[:,ind], row_linkage=row_linkage, col_linkage=col_linkage,cmap="Blues", standard_scale=1,yticklabels =False,xticklabels=np.array(tick))

	print np.shape(p2g_mat[:,ind].T),np.shape(tick)
	sns_plot.ax_row_dendrogram.set_visible(False)
	sns_plot.savefig(plot_result_dir+cancer+'_clustermap.pdf')

	plt.clf()
	show_leaf_counts = True
	ddata = augmented_dendrogram(linkage_matrix,'',
				   color_threshold=1,
				   p=topk,
				   truncate_mode='lastp',
				   show_leaf_counts=show_leaf_counts
				   )
	plt.title(cancer)
	plt.tight_layout()
	plt.savefig(plot_result_dir+cancer+'_dendrogram_large.pdf', transparent=True)

	sys.exit(-1)
	nc = np.shape(linkage_matrix)[0]
	gene_set_l = []
	i2gset = {}
	for i in range(topk):
		i2gset[i] = [i2g[ind[i]]]
	for i in range(nc):
		n1,n2 = linkage_matrix[i][0:2]
		n1_l = i2gset[n1]
		n2_l = i2gset[n2]
		i2gset[i+topk] = []
		for g in n1_l:
			i2gset[i+topk].append(g)
		for g in n2_l:
			i2gset[i+topk].append(g)

	i = term_id
	gene_set = set(i2gset[i+topk])
	valid_gene_set = set()
	for g in gene_set:
		valid_gene_set.add(g.lower())

	f2g_sc = read_all_func_data(valid_gene_set, G2G_obj)
	#sys.exit(-1)
	FN_obj = FindNeighbor(cname,ImproveNet_obj,stop_word_list=stop_word_list,exclude_edge_type = [],exclude_edges = [],include_genes=network_gene_list)
	g2score,gvec = FN_obj.CalNgh(G2G_obj,max_layer=max_layer,edge_wt_thres=edge_wt_thres,all_type_same_weight=False,use_direct_gene_edge=True,include_terms=include_terms)
	for gene_subnet_cutoff in [1e-7]:
		for gene_subnet_topk in [2]:
			node_set, edge_list, node_weight = FN_obj.GetSubNetwork(valid_gene_set,cutoff = gene_subnet_cutoff, topk = gene_subnet_topk,include_terms=include_terms,stop_word_list=stop_word_list)
			sign_term_set = query(gene_set,node_set,f2g_sc,G2G_obj,background_sc,kword=5)
			output_file = plot_result_dir + str(i)+'all'
			#plot_network_flow(output_file,cname,node_set,edge_list,node_weight, ImproveNet_obj.word_ct, ImproveNet_obj.word_type)
			filter_node_set,filter_edge_list = FN_obj.PruneSubNetwork(valid_gene_set, node_set, edge_list, include_terms=include_terms)
			output_file = plot_result_dir + str(i)+ '_' +  str(gene_subnet_cutoff) + '_' + str(gene_subnet_topk)
			filter_node_weight = {}
			for n in node_weight:
				if n in sign_term_set:
					filter_node_weight[n] = sign_term_set[n]
				else:
					filter_node_weight[n] = 1
			plot_network_flow(output_file,cname,filter_node_set,filter_edge_list,filter_node_weight, ImproveNet_obj.word_ct, ImproveNet_obj.word_type)
			write_network_cytoscape(output_file,filter_node_set,filter_edge_list,filter_node_weight,ImproveNet_obj.word_type)
	break


