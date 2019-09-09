import sys
import os
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
import pandas as PD
import collections
import numpy as np
from src.utils.evaluate.evaluate import evaluate_vec
from src.datasets.Gene2CellLine import Gene2CellLine
from src.datasets.DrugResponse import DrugResponse
from src.plot.ScatterPlot import ScatterPlot
from src.models.pathway_enrichment.PathwayEnrichment import PathwayEnrichment
from src.models.drug_response_prediction.target_based_prediction import TargetBasedPrediction
from src.models.drug_response_prediction.supervised_prediction import SupervisedPrediction,MyCrossValidation
#from src.datasets.SubGraph import SubGraph
from scipy import stats

class PathwayDrugResponse():

	def __init__(self,dataset='ccle'):
		self.ReadData(dataset)

	def ReadData(self,dataset):
		if dataset == 'ccle':
			drug_data_file = 'data/drug/ctrp/'
			self.dr_obj = DrugResponse(drug_data_file)
			self.g2c_obj = Gene2CellLine(PD.read_table('data/cell_line_data/CCLE_Expression_Entrez_2012-09-29.gct',
													 skiprows =2,index_col =1).drop(['Name'],axis=1),tissue_file='data/cell_line_data/CCLE_sample_info_file_2012-10-18.txt')
		else:
			drug_data_file = 'data/drug/gdsc/'
			self.dr_obj = DrugResponse(drug_data_file,drug_auc_file = 'label_auc_mapped.txt')
			self.g2c_obj = Gene2CellLine(PD.read_table('data/cell_line_data/gdsc_gep.txt'))

		fin = open(repo_dir+'data/cell_line_data/'+dataset+'_mutation.txt')
		self.c2g_mut = {}
		for line in fin:
			w = line.strip().split('\t')
			#print line
			c = w[1].split('_')[0]
			g = w[0].split('_')[0]
			#print c,g
			if c not in self.c2g_mut:
				self.c2g_mut[c] = {}
			self.c2g_mut[c][g] = 1
		fin.close()

	def CheckPvalue(self,drug,pathway):
		c2i = {}
		drug = drug.upper()
		for i in range(len(self.g2c_obj.g2c_cname)):
			c2i[self.g2c_obj.g2c_cname[i].split('_')[0]] = i

		#for nmute_cut in range(0,1):#len(pathway)):
		nmute_cut = 0
		pos = []
		neg = []
		for c in self.c2g_mut:
			if c not in c2i or c not in self.dr_obj.d2c[drug]:
				continue
			nmute = 0.
			for g in pathway:
				if g.upper() in self.c2g_mut[c]:
					nmute+=1
			if nmute>nmute_cut:
				pos.append(self.dr_obj.d2c[drug][c])
			else:
				neg.append(self.dr_obj.d2c[drug][c])
		if len(pos)<=1 or len(neg)<=1:
			#print len(pos),len(neg)
			return
		tstat, pv = stats.ranksums(pos, neg)
		return np.array(pos),np.array(neg), pv
			#print len(pathway),nmute_cut,pv,len(pos),len(neg)

	def CheckAnnovaPvalue(self,drug,pathway):

		c2i = {}
		drug = drug.upper()
		for i in range(len(self.g2c_obj.g2c_cname)):
			c2i[self.g2c_obj.g2c_cname[i].split('_')[0]] = i

		#for nmute_cut in range(0,1):#len(pathway)):
		tissue_set = np.unique(self.g2c_obj.c2tissue.values())

		nmute_cut = 0
		pos = []
		neg = []
		pos2 = []
		pos3 = []
		for c in self.c2g_mut:
			if c not in c2i or c not in self.dr_obj.d2c[drug]:
				continue
			nmute = 0.
			for g in pathway:
				if g.upper() in self.c2g_mut[c]:
					nmute+=1
			if nmute==1:
				pos.append(self.dr_obj.d2c[drug][c])
			elif nmute == 2:
				pos2.append(self.dr_obj.d2c[drug][c])
			elif nmute >=3 :
				pos3.append(self.dr_obj.d2c[drug][c])
			else:
				neg.append(self.dr_obj.d2c[drug][c])
		#if len(pos)<=1 or len(neg)<=1:
			#print len(pos),len(neg)
		#	return
		tstat, pv = stats.f_oneway(pos, neg, pos2,pos3)
		#tstat, pv = stats.ranksums(pos, neg)
		#seq = [pos2,pos,neg]
		#if pv<0.05:
		#print len(pos2),len(pos),len(neg)
		print len(pos),len(neg), len(pos2),len(pos3),pv
		seq = [pos3,pos2,pos,neg]
		return seq,pv

			#print len(pathway),nmute_cut,pv,len(pos),len(neg)





