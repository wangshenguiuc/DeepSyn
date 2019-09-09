import argparse
import re
import os
import random
import tarfile
import torch
import urllib
import numpy as np
from sklearn import metrics
from scipy import stats

class TextClassify():

	def __init__(self):
		self.w2i = {}

	def train(self,clf,text_data,feat_type='word'):
		self.w2i = {}
		self.i2w = {}
		self.nword = 0
		self.feat_type = feat_type
		wct = {}
		for text,label in text_data:
			wl = text.split(' ')
			for w in wl:
				wct[w] = wct.get(w,0) + 1

		for cutoff in [0]:
			self.w2i = {}
			self.i2w = {}
			self.nword = 0
			for text,label in text_data:
				wl = text.split(' ')
				for w in wl:
					if w not in self.w2i and w!='' and wct[w] > cutoff:
						self.w2i[w] = self.nword
						self.nword += 1
			if self.nword <3000:
				break

		for w in self.w2i:
			self.i2w[self.w2i[w]] = w
		trainX = []
		trainY = []
		for text,label in text_data:
			feat = self.get_feature(text)
			trainX.append(feat)
			trainY.append(int(label))
		trainX = np.array(trainX)
		trainY = np.array(trainY)
		pos = np.where(trainY==1)[0]
		neg = np.where(trainY==0)[0]
		self.clf = clf
		self.clf.fit(trainX, trainY)




	def predict_prob_fast(self, text):
		feat = [self.get_test_feature(text)]
		sc = self.clf.intercept_[0]
		for f in self.feat_list:
			sc += self.clf.coef_[0][f] * feat[0][f]
		prob = 1 / (1 + np.exp(-1*sc))
		return prob

	def get_keyword_ratio_based(self, nword):
		para = self.feat_wt*-1
		word_ind = np.argsort(para)[:nword]
		keyw = set()
		for w in word_ind:
			keyw.add(self.i2w[w])
		return keyw

	def get_keyword(self, nword):
		para = self.clf.coef_[0]*-1
		word_ind = np.argsort(para)[:nword]
		keyw = set()
		for w in word_ind:
			keyw.add(self.i2w[w])
		return keyw

	def get_test_feature(self,text):

		wl = text.split(' ')
		example = np.zeros(self.nword)
		for w in wl:
			if w not in self.f2i:
				continue
			example[self.f2i[w]] += 1
		return example

	def get_feature(self,text):
		if self.feat_type == 'word':
			return text
		else:
			wl = text.split(' ')
			example = np.zeros(self.nword)
			for w in wl:
				if w not in self.w2i:
					continue
				example[self.w2i[w]] += 1
		return example

	def predict(self, feat):
		# train or predict
		label = self.clf.predict([feat])[0]
		return label

	def predict_proba(self, feat):
		# train or predict
		#prob = self.clf.predict_proba(feat)[0][1]
		prob = self.clf.predict_proba([feat])[0]
		return prob

	def evaluate(self,test_text_data,pos_label=1):
		pred_l = []
		truth_l = []
		prob_l = []
		for text,label in test_text_data:
			feat = self.get_feature(text)
			pred_l.append(self.predict(feat))
			prob_l.append(self.predict_proba(feat)[pos_label])
			truth_l.append(int(label))
		pred_l = np.array(pred_l)
		truth_l = np.array(truth_l)
		prob_l = np.array(prob_l)
		cor = np.where(pred_l==truth_l)[0]
		acc = len(cor) * 1.0 / len(truth_l)
		sp = stats.spearmanr(prob_l,truth_l)[0]
		fpr,tpr,ths = metrics.roc_curve(truth_l, prob_l, pos_label = pos_label)
		auc = metrics.auc(fpr, tpr)
		#print model_name,truth_l[:5],prob_l[:5],pred_l[:5],acc,auc,len(cor),sp
		return auc
