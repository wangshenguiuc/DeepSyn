import argparse
import re
import os
import random
import tarfile
import torch
import urllib
import numpy as np
from sklearn import metrics


class TextClassify():

	def __init__(self):
		self.w2i = {}

	def train(self,clf,text_data):
		self.w2i = {}
		self.i2w = {}
		self.nword = 0
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
			example = self.get_feature(text)
			trainX.append(example)
			trainY.append(int(label))
		trainX = np.array(trainX)
		trainY = np.array(trainY)
		pos = np.where(trainY==1)[0]
		neg = np.where(trainY==0)[0]
		self.feat_wt = np.zeros(self.nword)
		for i in range(self.nword):
			self.feat_wt[i] = (np.sum(trainX[pos,i]) * 1.0 / len(pos) ) /(np.sum(trainX[neg,i]) * 1.0 / len(neg) )

		self.clf = clf
		self.clf.fit(trainX, trainY)


		self.feat_list = np.nonzero(self.clf.coef_)[1]
		self.f2i = {}
		self.i2f = {}
		for i in self.feat_list:
			w = self.i2w[i]
			self.f2i[w] = i#self.w2i[w]
			self.i2f[i] = w


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
		wl = text.split(' ')
		example = np.zeros(self.nword)
		for w in wl:
			if w not in self.w2i:
				continue
			example[self.w2i[w]] += 1
		return example

	def predict(self, sent):
		# train or predict
		label = self.clf.predict([self.get_feature(sent)])[0]
		return label

	def predict_prob(self, sent):
		# train or predict
		feat = [self.get_feature(sent)]
		#prob = self.clf.predict_proba(feat)[0][1]
		prob = self.clf.predict_proba(feat)[0][1]
		return prob

	def evaluate(self,test_text_data):
		pred_l = []
		truth_l = []
		prob_l = []
		print 'start evaluate--------------------------'
		for text,label in test_text_data:
			pred_l.append(self.predict(text))
			prob_l.append(self.predict_prob(text))
			truth_l.append(int(label))
		pred_l = np.array(pred_l)
		truth_l = np.array(truth_l)
		prob_l = np.array(prob_l)
		cor = np.where(pred_l==truth_l)[0]
		acc = len(cor) * 1.0 / len(truth_l)
		fpr,tpr,ths = metrics.roc_curve(truth_l, prob_l, pos_label = 1)
		auc = metrics.auc(fpr, tpr)
		return auc
