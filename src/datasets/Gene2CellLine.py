import numpy as np
import pandas as PD
#from repo.src.abstract_dataset import AbstractDataset
from numpy import corrcoef



class Gene2CellLine():
	def impute(self,a):
		col_mean = np.nanmean(a, axis=0)
		inds = np.where(np.isnan(a))
		a[inds] = np.take(col_mean, inds[1])
		return a

	def remove_tissue(self,cl):
		cl_new = []
		for i,c in enumerate(cl):
			if i==0 and c=='':
				continue
			w = c.split('_')
			cn = w[0]
			#for i in range(1,len(w)-1):
			#	cn += '_' + w[i]
			#print c,cn
			cl_new.append(cn.upper())
		return cl_new

	def cal_corr(self):
		self.g2g_cor = corrcoef(self.g2c_feat)
		return self.g2g_cor

	def read_tissue_file(self,tissue_file):
		self.c2tissue = {}
		fin = open(tissue_file)
		head = fin.readline().split('\t')
		site_ind = head.index('Site Primary')
		for line in fin:
			w = line.lower().strip().split('\t')
			for c in self.g2c_cname:
				cn = c.split('_')[0].lower()
				if cn==w[0] or cn==w[1] or cn==w[0].split('_')[0] or cn==w[1].split('_')[0]:
					self.c2tissue[cn] = w[site_ind]

	def __init__(self, df,impute=True,flip=False,tissue_file=''):
		if df == 'ctrp':
			df = PD.read_table('data/cell_line_data/CCLE_Expression_Entrez_2012-09-29.gct',skiprows =2,index_col =1).drop(['Name'],axis=1)
		elif df == 'gdsc':
			df = PD.read_table('data/cell_line_data/gdsc_gep.txt')
		elif df == 'dream_combo':
			df = PD.read_table('data/drug/dream_drug_combo/raw_data/dcombi/Sanger_molecular_data/gex.csv',delimiter=',')
		elif df == 'oneal_combo':
			df = PD.read_table('data/drug/ONeal_drug_combo/raw_data/oneal_gex.txt',delimiter='\t',index_col =0).transpose()

		if impute:
			self.g2c_feat = self.impute(df.values)
		if flip:
			self.g2c_feat = self.g2c_feat * -1
		self.g2i = {}
		self.i2g = {}
		self.c2i = {}
		self.g2c_gname =list(df.index)
		for i in range(len(self.g2c_gname)):
			self.g2i[self.g2c_gname[i]] = i
			self.i2g[i] = self.g2c_gname[i]
		
		self.g2c_cname = self.remove_tissue(list(df.columns.values))
		for i in range(len(self.g2c_cname)):
			self.c2i[self.g2c_cname[i]] = i
		print self.c2i
		assert(np.shape(self.g2c_feat)[0] == len(self.g2c_gname))
		assert(np.shape(self.g2c_feat)[1] == len(self.g2c_cname))

		if tissue_file!='':
			self.read_tissue_file(tissue_file)
