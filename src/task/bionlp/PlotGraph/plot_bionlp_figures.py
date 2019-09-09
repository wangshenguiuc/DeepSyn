import sys
import os
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
sys.path.append(repo_dir)
os.chdir(repo_dir)

from src.datasets.Gene2CellLine import Gene2CellLine
from src.datasets.DrugResponse import DrugResponse
from src.plot.ScatterPlot import ScatterPlot
from src.models.pathway_enrichment.PathwayEnrichment import PathwayEnrichment
from src.models.drug_response_prediction.target_based_prediction import TargetBasedPrediction
from src.models.drug_response_prediction.supervised_prediction import SupervisedPrediction,MyCrossValidation
#from src.datasets.SubGraph import SubGraph

from src.utils.evaluate.evaluate import evaluate_2D_dict,evaluate_1D_dict
import operator
import collections
import pandas as PD
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
#matplotlib.use('agg')
matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt
from scipy import stats
import math
from scipy import stats
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time
from sklearn.decomposition import PCA

def plot_waterfall(gscore,output_file,title=''):
	ngene = len(gscore)
	f = plt.figure()
	x = np.arange(ngene)
	y = -1*np.log10(gscore)

	plt.plot(x,y,'b^-')
	plt.xlabel('ranking')
	plt.ylabel('-1*log(gene score)')
	plt.title(title)
	#plt.show()
	f.savefig(output_file+'.pdf', bbox_inches='tight')
	f.savefig(output_file+'.png', bbox_inches='tight')

def plot_subnet_anova_boxplot(output_file,seq,title='subnet'):
	#tstat, pv = stats.ranksums(pos_v, neg_v)
	# Create a figure instance
	fig = plt.figure(1, figsize=(9, 6))

	# Create an axes instance
	#ax = fig.add_subplot(111)
	f = plt.figure()
	# Create the boxplot
	bp = plt.violinplot(seq)
	#plt.xticklabels(['mutated','not mutated'])
	plt.ylabel('drug response')
	plt.title(title)
	# Save the figure
	#plt.xticks([1, 2], ['mutated(n='+str(len(pos_v))+')', 'not mutated(n='+str(len(neg_v))+')'])
	f.savefig(output_file+'.png', bbox_inches='tight')
	f.savefig(output_file+'.pdf', bbox_inches='tight')

def plot_subnet_boxplot(output_file,pos_v,neg_v,title='subnet'):
	tstat, pv = stats.ranksums(pos_v, neg_v)
	# Create a figure instance
	fig = plt.figure(1, figsize=(9, 6))

	# Create an axes instance
	#ax = fig.add_subplot(111)
	f = plt.figure()
	# Create the boxplot
	bp = plt.boxplot([pos_v,neg_v])
	#plt.xticklabels(['mutated','not mutated'])
	plt.ylabel('drug response')
	plt.title(title+' '+str(pv))
	# Save the figure
	#plt.xticks([1, 2], ['mutated(n='+str(len(pos_v))+')', 'not mutated(n='+str(len(neg_v))+')'])
	f.savefig(output_file+'.png', bbox_inches='tight')
	f.savefig(output_file+'.pdf', bbox_inches='tight')


def plot_gene_scatter(drug,gene,output_file):
	g2c_obj = {}
	drug = drug.upper()
	gdsc_data_file = 'data/cell_line_data/gdsc_gep.txt'
	g2c_obj['gdsc'] = Gene2CellLine(PD.read_table(gdsc_data_file))

	g2c_obj['ccle'] = Gene2CellLine(PD.read_table('data/cell_line_data/CCLE_Expression_Entrez_2012-09-29.gct',
												 skiprows =2,index_col =1).drop(['Name'],axis=1))
	dr_obj = {}
	drug_data_file = 'data/drug/ctrp/'
	dr_obj['ccle'] = DrugResponse(drug_data_file)

	drug_data_file = 'data/drug/gdsc/'
	dr_obj['gdsc'] = DrugResponse(drug_data_file,drug_auc_file = 'label_auc_mapped.txt')

	for method in ['gdsc','ccle']:
		c2i = {}
		for i in range(len(g2c_obj[method].g2c_cname)):
			c2i[g2c_obj[method].g2c_cname[i].split('_')[0]] = i
		gid = g2c_obj[method].g2c_gname.index(gene)
		if drug not in dr_obj[method].d2c:
			continue
		#print dr_obj[method].d2c[drug]
		v1 = []
		v2 = []
		for c in dr_obj[method].d2c[drug]:
			if c not in c2i:
				#print c
				continue
			v1.append(dr_obj[method].d2c[drug][c])
			cid = c2i[c]
			v2.append(g2c_obj[method].g2c_feat[gid, cid])
		v1 = np.array(v1)
		v2 = np.array(v2)
		#print v1,v2
		cor, pv = stats.spearmanr(v1, v2)
		if np.abs(cor)<0.1:
			continue
		#print drug,gene,cor,pv, stats.pearsonr(v1, v2)
		plt.clf()
		f = plt.figure()
		plt.scatter(v1, v2, c="g", alpha=0.5)
		plt.xlabel("Drug response")
		plt.ylabel("Expression")
		cor = '%.2f' % (float(cor) )
		plt.title('Spearman: '+cor)
		f.savefig(output_file+method+'.png', bbox_inches='tight')
		f.savefig(output_file+method+'.pdf', bbox_inches='tight')






