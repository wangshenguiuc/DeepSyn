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
from adjustText import adjust_text
import seaborn as sns
from scipy import stats
from src.utils.evaluate.evaluate import evaluate_vec
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from src.models.pathway_enrichment.PathwayEnrichment import PathwayEnrichment
from utils import *
import parula as par

def pv2cat(pv):
	if pv>0.05:
		return 1.
	if pv>0.01:
		return 2.
	if pv>0.005:
		return 3.
	if pv>0.001:
		return 4.
	return 5

def augmented_dendrogram(linkage_matrix,base_term,our_term,color_threshold=1,nnode=0,p=2,truncate_mode='lastp',show_leaf_counts=True,gene_label=[]):
	ddata = dendrogram(linkage_matrix,color_threshold=color_threshold,p=p,truncate_mode=truncate_mode,show_leaf_counts=show_leaf_counts,
	leaf_rotation=90, leaf_font_size=8, labels=gene_label)
	y_l = []
	for y in ddata['dcoord']:
		y_l.append(y[1])
	y_l = np.array(y_l)
	y_ind = np.argsort(y_l)
	#print y_ind
	ranks = np.empty_like(y_ind)
	ranks[y_ind] = np.arange(len(y_l))
	#print ranks

	k=-1
	texts = []
	for i, d in zip(ddata['icoord'], ddata['dcoord']):
		x = 0.5 * sum(i[1:3])
		y = d[1]
		k+=1
		ind = ranks[k] + nnode - p
		b_term,b_pv = base_term[ind]
		if b_term=='Not computed':
			continue
		o_term,o_pv = our_term[ind]
		b_cat = pv2cat(b_pv)
		o_cat = pv2cat(o_pv)
		b_mark_size = b_cat**2 * 3
		o_mark_size = o_cat**2 * 3
		print b_term,b_mark_size
		print o_term,o_mark_size
		if b_cat>1:
			plt.plot(x, y, 'o',markersize=b_mark_size,markeredgewidth=1.,markeredgecolor='white',markerfacecolor='none')
			texts.append(plt.text( x, y, b_term, ha='center', wrap=True,color='white',fontsize=5,rotation=90))
		if o_cat>1:
			plt.plot(x, y, 'o',markersize=o_mark_size,markeredgewidth=1.,markeredgecolor='pink',markerfacecolor='none')
			texts.append(plt.text( x, y, o_term, ha='center', wrap=True,color='pink',fontsize=5,rotation=90))
			#plt.plot(x, y, 'ro',markeredgecolor='white',markerfacecolor='white')


	adjust_text(texts,  only_move={'points':'y', 'text':'y', 'objects':'y'})#, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
	return ddata

def query(gene_set,candidate_term,f2g_mat,background_sc,kword=5,stop_word = []):
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
	sign_term_set = []
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
			sign_term_set.append((f,pv))
		if len(sign_term_set)==3:
			break
	if len(sign_term_set) == 0:
		sign_term_set = [(best_term,best_pv)]
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




parula_cmap = par.parula_map

viridis_rgb = []
magma_rgb = []
parula_rgb = []
norm = matplotlib.colors.Normalize(vmin=0, vmax=255)

for i in range(0, 255):
       k = matplotlib.colors.colorConverter.to_rgb(parula_cmap(norm(i)))
       parula_rgb.append(k)

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k*h)[:3])*255)
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

parula = matplotlib_to_plotly(parula_cmap, 255)

dataset = 'everything'
ImproveNet_obj,ImproveNet_obj_baseline,stop_word_list = read_ImprovedNet_obj(dataset=dataset,min_freq_cutoff=4,max_ngh=100)

G2G_obj,network_gene_list = read_network_data(net_file_l = ['data/network/human/string_integrated.txt'])
ngene = len(G2G_obj.g2i)

stop_word_list.add('c6 glioma cell')
stop_word_list.add('stem cell phenotype')
stop_word_list.add('gbm tumors')
stop_word_list.add('glioma cell growth')
stop_word_list.add('primary brain tumor')
stop_word_list.add('vegf family')
stop_word_list.add('vegf pathway')
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
stop_word_list.add('histone deacetylation')
stop_word_list.add('induced g1 arrest')
stop_word_list.add('methyl-cpg binding')
stop_word_list.add('apoptosome')
stop_word_list.add('proteasomal inhibition')



include_terms = set()
include_terms.add('cell autophagy')
include_terms.add('vessel maturation')
include_terms.add('astrocyte differentiation')
include_terms.add('catabolic process')
include_terms.add('glial cell migration')
include_terms.add('nuclear body')


pvalue_dec = 1e6
result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/function_score/pvalue/'
fin = open(result_dir+str(int(pvalue_dec)))
background_sc = []
for line in fin:
	w = line.strip()
	background_sc.append(float(w))
fin.close()
background_sc = np.array(background_sc)

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

for ci,exp_file in enumerate(exp_file_l):
	if ci%total_pid != pid and total_pid>1 :
		continue
	cancer = exp_file.replace('.txt','')
	if cancer not in c2n:
		continue
	cancer_name = c2n[cancer]
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
	gene_set = ["tmsb15a", "vsnl1", "sox10", "st8sia3", "ca10", "luzp2", "nnat", "cndp1", "myt1", "cdkn2a", "ppan-p2ry11", "vstm2a", "slc17a7", "ina", "pcdh15", "kdm5d", "ddx3y", "stmn2", "pcsk2", "eif1ay", "col11a1", "rps4y1", "dcx", "f13a1", "mog", "actl6b", "gpr17", "cplx2", "zfy", "tmsb4y", "nlgn4y", "fbn3", "postn", "usp9y", "vipr2", "ltf", "klk6", "mobp", "uty", "col20a1"]

	ind  = []
	ind_name = []
	for g in gene_set:
		ind.append(g2i[g.upper()])
		ind_name.append(g.upper())
	ind = np.array(ind)
	ind_name = np.array(ind_name)
	print len(ind)
	topk = len(ind)
	#ind = np.argsort(gstd*-1)[:min(topk,ngene)]

	tick = []
	select_gene_set = set()
	for id in ind:
		select_gene_set.add(i2g[id])
		tick.append(i2g[id])
	f2g_sc = read_all_func_data(select_gene_set, G2G_obj)

	PE = PathwayEnrichment(pathfile_l=['data/pathway/gene_ontology.txt'],gene_set = set(g2i.keys()))

	linkage_matrix = linkage(p2g_mat[:,ind].T, "ward")

	nc = np.shape(linkage_matrix)[0]
	base_term = {}
	our_term = {}
	i2gset = {}
	for i in range(topk):
		i2gset[i] = [i2g[ind[i]]]
	fout = open(plot_result_dir+'detail.txt','w')
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
		if i<nc-max_clst:
			base_term[i] = ('Not computed',1)
			our_term[i] = ( 'Not computed',1)
			continue
		fout.write(str(i)+'\t')
		sign_p = PE.enrich(gene_set,ngene=ngene)[0]
		base_pv = 1
		our_pv = 1
		base_name =''
		if len(sign_p) > 0:
			minpv = np.min(sign_p.values())

			for nm in sign_p:
				if minpv == sign_p[nm]:
					base_name = nm
			base_pv = minpv
		base_term[i] = (base_name, base_pv)

		valid_gene_set = set()
		for g in gene_set:
			if g in G2G_obj.g2i:
				valid_gene_set.add(g.lower())

		FN_obj = FindNeighbor(cancer_name,ImproveNet_obj,stop_word_list=stop_word_list,exclude_edge_type = [],exclude_edges = [],include_genes=network_gene_list)
		g2score,gvec = FN_obj.CalNgh(G2G_obj,max_layer=max_layer,edge_wt_thres=edge_wt_thres,all_type_same_weight=False,use_direct_gene_edge=True)
		gene_subnet_cutoff = 1e-7
		gene_subnet_topk = 2
		node_set, edge_list, node_weight = FN_obj.GetSubNetwork(valid_gene_set,cutoff = gene_subnet_cutoff, topk = gene_subnet_topk,stop_word_list=stop_word_list,include_terms=include_terms)
		term_set = set()
		graph_gene_list = []
		for n in node_set:
			if ImproveNet_obj.word_type[n].split('_')[1] == 'gene':
				graph_gene_list.append(n)
			elif n!=cancer_name:
				term_set.add(n)
		#if i==83:
		#	node_set, edge_list, node_weight = FN_obj.GetSubNetwork(valid_gene_set,cutoff = gene_subnet_cutoff, topk = gene_subnet_topk,stop_word_list=stop_word_list,include_terms=include_terms)
		##	node_set,edge_list = FN_obj.PruneSubNetwork(valid_gene_set, node_set, edge_list, include_terms=include_terms)
		for g in valid_gene_set:
			fout.write(g+'\t'+str(g in graph_gene_list)+'\t')
		fout.write('\n')
		sign_term_set = query(gene_set,term_set,f2g_sc,G2G_obj,background_sc,kword=5,stop_word=stop_word_list)

		best_term = ''
		our_term[i] = ('None',0.05)
		for term,pv in sign_term_set:
			best_term += term + '(%.2e)' % pv + '\n'
			if pv<our_pv:
				our_pv = pv
		our_term[i] = (best_term,our_pv)
		fout.write(best_term+'\n')

	plt.clf()
	plt.style.use('dark_background')
	fout.close()
	show_leaf_counts = True
	ddata = augmented_dendrogram(linkage_matrix,base_term=base_term,our_term=our_term,nnode = topk,
				   color_threshold=1,
				   p=topk,
				   truncate_mode='lastp',
				   show_leaf_counts=show_leaf_counts,gene_label=ind_name
				   )
	plt.title(cancer)
	plt.tight_layout()
	plt.savefig(plot_result_dir+cancer+'_dendrogram.pdf')

	plt.clf()
	plt.style.use('dark_background')
	row_linkage = linkage(p2g_mat[:,ind].T, "ward")
	col_linkage = linkage(p2g_mat[:,ind], "ward")
	sns.set(style="ticks", context="talk",font_scale=0.4)
	sns.axes_style("dark")
	#sns.yaxis.tick_right()
	parula_cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
	sns_plot = sns.clustermap(p2g_mat[:,ind].T, row_linkage=row_linkage, col_linkage=col_linkage,cmap=parula_cmap, standard_scale=1,xticklabels =False,yticklabels=np.array(tick))#coolwarm

	sns_plot.ax_row_dendrogram.set_visible(False)
	#sns_plot.ax_column_dendrogram.set_visible(False)
	sns_plot.savefig(plot_result_dir+cancer+'_clustermap.pdf')
	#sys.exit(-1)
