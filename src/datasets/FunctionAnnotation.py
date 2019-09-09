import numpy as np
import collections
from scipy import sparse
from src.models.drug_response_prediction.supervised_prediction import MyCrossValidation
from sklearn.linear_model import LogisticRegression

class FunctionAnnotation():

	def __init__(self, annotation_file, GO_net,  GO_name_file = 'data/function_annotation/GO2name.txt',select_GO_set = []):
		self.f2g = {}
		self.select_GO_set = select_GO_set
		self.ReadAnnotation(annotation_file, GO_net)
		self.ReadGO2Gname(GO_name_file)

	def ReadGO2Gname(self,GO_name_file):
		fin = open(GO_name_file)
		self.GO2name  ={}
		self.name2GO  ={}
		self.GO2cat = {}
		for line in fin:
			w  = line.strip().split('\t')
			if len(w) < 2:
				continue
			self.GO2name[w[0]] = w[1]
			self.name2GO[w[1]] = w[0]
			self.GO2cat[w[0]] = w[2]
		fin.close()

	def add(self, f, g):
		if f not in self.f2g:
			self.f2g[f] = set()
		self.f2g[f].add(g)

	def get_parents(self, GO_net, g):
		term_valid = set()
		ngh_GO = set()
		ngh_GO.add(g)
		while len(ngh_GO) > 0:
			for GO in list(ngh_GO):
				for GO1 in GO_net[GO]:
					ngh_GO.add(GO1)
				ngh_GO.remove(GO)
				term_valid.add(GO)
		return term_valid

	def ReadAnnotation(self, annotation_file, GO_net):
		#print file_name
		fin = open(annotation_file)
		for line in fin:
			if line.startswith('!'):
				continue
			w = line.strip().split('\t')
			g = w[2].lower()
			f = w[4]
			if len(self.select_GO_set) > 0 and f not in self.select_GO_set:
				continue
			self.add(f,g)
			pat = self.get_parents(GO_net,f)
			for ff in pat:
				self.add(ff,g)
		self.i2f = {}
		self.f2i = {}
		c = 0
		for f in self.f2g:
			self.i2f[c] = f
			self.f2i[f] = c
			c += 1

	def PredictAnnotation(self, g2feat, g2i, clf=LogisticRegression()):
		g2feat = np.array(g2feat)
		ngene = len(g2i)
		nfunc = len(self.f2g)
		label_l = np.zeros((nfunc,ngene))
		avg_auc = []
		for f in self.f2g:
			if f not in self.GO2name:
				continue
			feat_l = []
			for g in g2i.keys():
				gid = g2i[g]
				feat = g2feat[gid,:]
				label_l[self.f2i[f],gid] = int(g in self.f2g[f])
				feat_l.append(feat)
			if np.sum(label_l[self.f2i[f],:])<=3:
				continue
			CV_obj = MyCrossValidation(clf,feat_l,label_l[self.f2i[f],:],nfold=3,nrepeat=10)
			print self.GO2name[f], np.sum(label_l[self.f2i[f],:]), CV_obj.train_score['auroc'],CV_obj.test_score['auroc'],np.mean(avg_auc)
			avg_auc.append(CV_obj.test_score['auroc'])
		return np.mean(avg_auc)

