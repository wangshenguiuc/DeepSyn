import numpy as np
import collections
from src.utils.evaluate.evaluate import evaluate_2D_dict,evaluate_1D_dict
import multiprocessing
import sys
import os

class TargetBasedPrediction():

	def __init__(self,g2c,dr_obj,combine_target_method='max',verbal=True):
		self.verbal = True
		self.g2c_feat = g2c.g2c_feat
		self.g2c_cname = g2c.g2c_cname
		self.g2c_gname = g2c.g2c_gname
		self.c2cid = {}
		for c in g2c.g2c_cname:
			self.c2cid[c] = g2c.g2c_cname.index(c)
		self.g2gid = {}
		for g in g2c.g2c_gname:
			self.g2gid[g] = g2c.g2c_gname.index(g)
		self.g2c_cname.index(c)
		self.d2g = dr_obj.d2g
		self.d2c = dr_obj.d2c
		self.combine_target_method = combine_target_method
		self.predict()
		self.auc,self.pear,self.spear,self.auc_d,self.pear_d,self.spear_d = evaluate_2D_dict(self.d2c_pred,self.d2c)

	def combine_score(self,sc_l):
		if len(sc_l)==0:
			return np.nan,-1
		sc_l = np.array(sc_l)
		if self.combine_target_method=='max':
			sc = np.max(sc_l)
			ind = np.argmax(sc_l)
		elif self.combine_target_method=='min':
			sc = np.min(sc_l)
			ind = np.argmin(sc_l)
		elif self.combine_target_method=='mean':
			sc = np.mean(sc_l)
			ind = np.argmean(sc_l)
		else:
			raise ValueError('wrong combine target method')
		return sc, ind

	def predict(self,verbal = False):
		g2c_cname_set = set(self.g2c_cname)
		self.d2c_pred = collections.defaultdict(dict)
		self.d2c_tgt = collections.defaultdict(dict)
		ntotal = len(self.d2c)*1.0
		for ct,d in enumerate(self.d2c):
			for c in self.d2c[d]:
				if c not in g2c_cname_set:
					continue
				cid = self.g2c_cname.index(c)
				sc_l = []
				tlist = self.d2g[d].keys()
				for t in tlist:
					if t not in self.g2c_gname:
						continue
					gid = self.g2c_gname.index(t)
					sc = self.g2c_feat[gid, cid]
					sc_l.append(sc)
				if len(sc_l) == 0:
					continue
				sc, ind = self.combine_score(sc_l)
				self.d2c_tgt[d][c] = tlist[ind]
				if not np.isnan(sc):
					self.d2c_pred[d][c] = sc
			if ct%10==0 and verbal:
				print 'finished',ct*1.0/ntotal

	def mp_iterate_all_genes(self,d2g_d,pid,dset,min_support=10):

		ntotal = len(dset)
		g2c_cname_set = set(self.g2c_cname)
		for ct,d in enumerate(dset):
			for ii,g in enumerate(self.g2c_gname):
				if (not type(g) == str) and np.isnan(g):
					continue
				gid = self.g2gid[g]
				gs_d = {}
				cs_d = {}
				for c in self.d2c[d]:
					if c not in g2c_cname_set:
						continue
					cid = self.c2cid[c]
					gs_d[c] = self.g2c_feat[gid, cid]
					cs_d[c] = self.d2c[d][c]
				if len(cs_d)<min_support:
					continue
				auc,pear,spear = evaluate_1D_dict(gs_d,cs_d)
				d2g_d[(d,g)] = pear
			if pid==0 and ct%2==0:
				print 'finished',ct,'in',ntotal


	def iterate_all_genes(self, verbal =False,nproc=64):
		##iterate over all genes and find the score for each gene for each drug
		self.d2g_pred_pear = collections.defaultdict(dict)
		ntotal = len(self.d2c)*1.0
		manager = multiprocessing.Manager()
		d2g_d = manager.dict()
		print 'start running'
		p2chunk = {}
		for i in range(nproc):
			p2chunk[i] = set()
		for c,w in enumerate(self.d2c):
			p2chunk[c%nproc].add(w)
		jobs = []
		for i in range(nproc):
			p = multiprocessing.Process(target=self.mp_iterate_all_genes, args=(d2g_d,i,p2chunk[i]))
			jobs.append(p)
			p.start()
		for proc in jobs:
			proc.join()
		for (d,g) in d2g_d.keys():
			self.d2g_pred_pear[d][g] = d2g_d[(d,g)]


	def select_sign_genes(self, thres = 0.2,two_side=True):
		self.sign_gene_set = collections.defaultdict(set)
		size = np.array([])
		self.sign_gene_set_size = {}
		for d in self.d2g_pred_pear:
			self.sign_gene_set[d] = set()
			for g in self.d2g_pred_pear[d]:
				if two_side:
					sc = abs(self.d2g_pred_pear[d][g])
				else:
					sc = self.d2g_pred_pear[d][g]
				if sc > thres:
					self.sign_gene_set[d].add(g)
			size = np.append(size,len(self.sign_gene_set[d]))
		print 'size avg',np.min(size),np.mean(size),np.median(size),np.max(size)
		self.sign_gene_set_size = size
		return self.sign_gene_set

