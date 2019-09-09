import numpy as np
import collections
from src.utils.evaluate.evaluate import evaluate_2D_dict,evaluate_1D_dict,evaluate_vec
import multiprocessing
import sys
import os
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,KFold
from scipy import stats
import scipy
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
class MyCrossValidation():
	def __init__(self,clf,X,Y,nfold=3,nrepeat=1):
		X = np.array(X)
		Y = np.array(Y)
		is_classify = False
		if np.array_equal(Y, Y.astype(bool)):
			is_classify = True
		self.Ytest_sc_l = []
		self.Ytrain_sc_l = []
		self.train_score = {}
		self.test_score = {}
		Ytest_auc_l = []
		Ytrain_auc_l = []
		nsample,nfeat = np.shape(X)
		for i in range(nrepeat):
			if nfold!=-1:
				kf = KFold(nfold,True,random_state=i)
				kf.get_n_splits(X)
				train_test_iter = kf.split(X)
			else: # leave one out
				train_test_iter = []
				for i in range(nsample):
					#bef = [0:i]
					#aft = [i+1:nsample]
					train_index = bef.extend(aft)
					train_index = np.array(train_index)
					test_index = i
					train_test_iter.append(train_index,test_index)
			for train_index, test_index in train_test_iter:
				X_train, X_test = X[train_index], X[test_index]
				Y_train, Y_test = Y[train_index], Y[test_index]
				if nfeat>1:
					clf.fit(X_train,Y_train)
					if not is_classify:
						Y_train_pred = clf.predict(X_train)
						Y_test_pred = clf.predict(X_test)
						Ytrain_sc = scipy.stats.spearmanr(Y_train_pred, Y_train)[0]
						Ytest_sc = scipy.stats.spearmanr(Y_test_pred, Y_test)[0]
						self.Ytest_sc_l.append(Ytest_sc)
						self.Ytrain_sc_l.append(Ytrain_sc)
					else:
						if np.sum(Y_train)==0:
							continue
						Y_train_pred = clf.predict_proba(X_train)[:,1]
						Y_test_pred = clf.predict_proba(X_test)[:,1]
						Ytrain_sc = evaluate_vec(Y_train_pred,Y_train)[0]
						Ytest_sc = evaluate_vec(Y_test_pred,Y_test)[0]
						self.Ytest_sc_l.append(Ytest_sc)
						self.Ytrain_sc_l.append(Ytrain_sc)
				else:
					Y_train_pred = X_train[:,0]
					Y_test_pred = X_test[:,0]
					Ytrain_sc = scipy.stats.spearmanr(Y_train_pred, Y_train)[0]
					Ytest_sc = scipy.stats.spearmanr(Y_test_pred, Y_test)[0]
					self.Ytest_sc_l.append(Ytest_sc)
					self.Ytrain_sc_l.append(Ytrain_sc)
		if not is_classify:
			self.train_score['spearman'] = np.nanmean(self.Ytrain_sc_l)
			self.test_score['spearman'] = np.nanmean(self.Ytest_sc_l)
		else:
			self.train_score['auroc'] = np.nanmean(self.Ytrain_sc_l)
			self.test_score['auroc'] = np.nanmean(self.Ytest_sc_l)

	def evaluate_auc(self, y, pred):
		fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
		auc = metrics.auc(fpr, tpr)
		return auc


class SupervisedPrediction():

	def __init__(self,g2c,dr_obj,clf,param_grid):
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
		#clf = GridSearchCV(estimator=clf, param_grid=param_grid,n_jobs=1)
		self.predict(clf, verbal=False)


	def predict(self,clf, verbal = False):

		g2c_cname_set = set(self.g2c_cname)
		self.d2score = collections.defaultdict(dict)
		ntotal = len(self.d2c)*1.0
		for ct,d in enumerate(self.d2c):
			rsp_l = []
			feat_l = []
			for c in self.d2c[d]:
				if c not in g2c_cname_set:
					continue
				cid = self.g2c_cname.index(c)
				rsp_l.append(self.d2c[d][c])
				feat_l.append(self.g2c_feat[:, cid])
			rsp_l = np.array(rsp_l)
			feat_l = np.array(feat_l)
			cv_obj = MyCrossValidation(clf,feat_l,rsp_l)
			self.d2score[d] = cv_obj.test_score
			if ct%10==0 and verbal:
				print 'finished',ct*1.0/ntotal
		self.test_score = {}
		for tp in self.d2score[d]:
			sc_l = []
			for d in self.d2score:
				sc_l.append(self.d2score[d][tp])
			self.test_score[tp] = np.mean(sc_l)
