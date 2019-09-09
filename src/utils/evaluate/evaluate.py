import numpy as np
import logging
import os
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import scipy


def precision_at_k(pred,truth, k=10,pos_label=1):
	n_pos = np.sum(truth == pos_label)

	order = np.argsort(pred)[::-1]
	truth = np.take(truth, order[:k])
	n_relevant = np.sum(truth == pos_label)

	# Divide by min(n_pos, k) such that the best achievable score is always 1.0.
	return float(n_relevant) / min(k,len(truth))#min(n_pos, k)


def evaluate_vec(pred,truth):
	pred = np.array(pred)
	truth = np.array(truth)


	if set(np.unique(truth))==set([0,1]):
		fpr, tpr, thresholds = metrics.roc_curve(truth, pred, pos_label=1)
		auc = metrics.auc(fpr, tpr)
		auprc = metrics.average_precision_score(truth, pred)
		pear = 0
		spear = 0
		prec_at_k = precision_at_k(pred,truth)
	else:
		auc = 0.5
		auprc = 0
		pear = scipy.stats.pearsonr(pred, truth)[0]
		spear = scipy.stats.spearmanr(pred, truth)[0]
		prec_at_k = 0
	return auc,pear,spear,auprc,prec_at_k

def evaluate_1D_dict(pred,truth):
	pred_l = np.array([])
	truth_l = np.array([])
	for c in pred:
		if c not in truth:
			continue
		pred_l = np.append(pred_l,pred[c])
		truth_l = np.append(truth_l,truth[c])
	auc,pear,spear = evaluate_vec(pred_l,truth_l)
	return auc,pear,spear

def evaluate_2D_dict(pred,truth):
	auc_d = {}
	pear_d = {}
	spear_d = {}
	auc_l = np.array([])
	pear_l = np.array([])
	spear_l = np.array([])
	for x in pred:
		if x not in truth:
			continue
		auc,pear,spear = evaluate_1D_dict(pred[x],truth[x])
		auc_d[x] = auc
		pear_d[x] = pear
		spear_d[x] = spear
		auc_l = np.append(auc_l,auc)
		pear_l = np.append(pear_l,pear)
		spear_l = np.append(spear_l,spear)
	auc = np.mean(auc_l)
	pear = np.mean(pear_l)
	spear = np.mean(spear_l)
	return auc,pear,spear,auc_d,pear_d,spear_d
