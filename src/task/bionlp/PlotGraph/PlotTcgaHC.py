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
from adjustText import adjust_text
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
plt.switch_backend('agg')
plt.style.use('ggplot')

def augmented_dendrogram(linkage_matrix,gene_set_l,
				   color_threshold=1,
				   p=2,
				   truncate_mode='lastp',
				   show_leaf_counts=True):
	ddata = dendrogram(linkage_matrix,
				   color_threshold=color_threshold,
				   p=p,
				   truncate_mode=truncate_mode,
				   show_leaf_counts=show_leaf_counts)
	print ddata['ivl']
	y_l = []
	for y in ddata['dcoord']:
		y_l.append(y[1])
	y_l = np.array(y_l)
	y_ind = np.argsort(y_l)
	new_gene_set_l = []
	for k in y_ind:
		new_gene_set_l.append(gene_set_l[k])
	k=0

	texts = []
	for i, d in zip(ddata['icoord'], ddata['dcoord']):
		x = 0.5 * sum(i[1:3])
		y = d[1]
		plt.plot(x, y, 'ro')
		if new_gene_set_l[k].startswith('NET'):
			c=  'red'
		else:
			c= 'blue'
		texts.append(plt.text( x, y, new_gene_set_l[k], ha='center', wrap=True,color=c))
		k+=1
		print k,i,d
	adjust_text(texts,  only_move={'points':'y', 'text':'y', 'objects':'y'})#, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
	return ddata

def query(gene_set,f2g_mat,G2G_obj,background_sc,kword=5):
	term_set = []
	sc = {}
	for f in f2g_mat:
		sc[f] = 0.
		ngene_term = 0
		for g in gene_set:
			if g in f2g_mat[f]:
				sc[f]+= f2g_mat[f][g]
				ngene_term+=1
		if ngene_term==0:
			continue
		sc[f] /= ngene_term

		#print f,sc[f], pv[f],(len(np.where(background_sc>sc[f])[0]) + 1),len(background_sc)
	sorted_x = sorted(sc.items(), key=operator.itemgetter(1))
	sorted_x.reverse()
	for i in range(kword):
		f = sorted_x[i][0]
		pv = (len(np.where(background_sc>sc[f])[0]) + 1) * 1. / len(background_sc)
		term_set.append((f, pv))
		#print pv[sorted_x[i][0]]
	return term_set

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

G2G_obj,network_gene_list = read_network_data(net_file_l = ['data/network/human/string_integrated.txt'])

plot_result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/TCGA_expression/'
if not os.path.exists(plot_result_dir):
	os.makedirs(plot_result_dir)

pvalue_dec = 1e7
result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/function_score/pvalue/'
fin = open(result_dir+str(int(pvalue_dec)))
background_sc = []
for line in fin:
	w = line.strip()
	background_sc.append(float(w))
fin.close()
background_sc = np.array(background_sc)
topk = 100
expression_folder = 'data/TCGA/expression/'
exp_file_l = os.listdir(expression_folder)
max_clst = 20
for ci,exp_file in enumerate(exp_file_l):
	print exp_file
	cancer = exp_file.replace('.txt','')
	p2g_exp = collections.defaultdict(dict)
	fin = open('data/TCGA/expression/' + exp_file)
	p2i = {}
	g2i = {}
	i2p = {}
	i2g = {}
	for line in fin:
		p,g,v = line.upper().strip().split('\t')
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
	for i in ind:
		tick.append(i2g[i])

	select_gene_set = set()
	for id in ind:
		select_gene_set.add(i2g[id])
	f2g_sc = read_all_func_data(select_gene_set, G2G_obj)
	print len(f2g_sc)

	PE = PathwayEnrichment(pathfile_l=['data/pathway/gene_ontology.txt'],gene_set = set(g2i.keys()))

	linkage_matrix = linkage(p2g_mat[:,ind].T, "ward")
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
		gene_set = set(i2gset[i+topk])
		print gene_set
		sign_p = PE.enrich(gene_set,ngene=ngene)[0]
		term_set = query(gene_set,f2g_sc,G2G_obj,background_sc)
		if len(sign_p)!=0:
			minpv = np.min(sign_p.values())
			name = ''
			for nm in sign_p:
				if minpv == sign_p[nm]:
					name = nm + '\n%.2e' % minpv
		else:
			#name = ''
			#for g in gene_set:
			#	name+=g+' '
			#name = 'NA'
			name = 'NET:'+term_set[0][0] +  '\n%.2e' % term_set[0][1]
		gene_set_l.append(name)
		print i,name
	print nc
	print gene_set_l[-max_clst:]
	plt.clf()

	show_leaf_counts = True
	ddata = augmented_dendrogram(linkage_matrix,gene_set_l[-max_clst:],
				   color_threshold=1,
				   p=max_clst,
				   truncate_mode='lastp',
				   show_leaf_counts=show_leaf_counts
				   )
	plt.title(cancer)
	plt.tight_layout()
	plt.savefig(plot_result_dir+cancer+'_dendrogram.pdf')

	plt.clf()
	row_linkage = linkage(p2g_mat[:,ind].T, "ward")
	col_linkage = linkage(p2g_mat[:,ind], "ward")
	sns_plot = sns.clustermap(p2g_mat[:,ind].T, row_linkage=row_linkage, col_linkage=col_linkage,cmap="Blues", standard_scale=1,xticklabels=tick,yticklabels =False)
	sns_plot.savefig(plot_result_dir+cancer+'_clustermap.pdf')
	#sys.exit(-1)
