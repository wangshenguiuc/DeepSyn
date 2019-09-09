import multiprocessing
import sys
import random
import os
import pickle
repo_dir = '/data/cellardata/users/majianzhu/wangsheng/Sheng_repo/Sheng_repo/'
sys.path.append(repo_dir)
os.chdir(repo_dir)

from src.datasets.WordNet import parse_sentence
import operator
import collections
from src.models.text_classification.TextClassify import TextClassify
from sklearn.linear_model import LogisticRegression
import numpy as np

def predict_GO_term(GO,GO_term_dir,GO2text,negative_set):
	GO2keyword = {}
	nkeyword = 20

	GO_predict_dir = '/cellar/users/majianzhu/Data/wangsheng/Sheng_repo/data/Pubmed/predict_GO'
	text_file = GO_predict_dir + GO.replace('/','_').replace(' ','_')+'.txt'

	if os.path.isfile(text_file):
		continue
	if not os.path.exists(GO_predict_dir):
		os.makedirs(GO_predict_dir)
	nmax_train_data = 10000
	nmin_train_data = 10000

	GO_term_file = GO_term_dir + GO.replace('/','_').replace(' ','_') +'.pkl'
	train_data = []
	if len(GO2text[GO]) > nmax_train_data:
		select_sent = np.random.choice(GO2text[GO].keys(), nmax_train_data, replace=False)
		for t in select_sent:
			train_data.append((t.replace(GO,''),1))
	else:
		for t in GO2text[GO]:
			train_data.append((t.replace(GO,''),1))

	ntrain_data = min(max(nmin_train_data,len(train_data)*9), len(negative_set))
	train_data_neg = np.random.choice(negative_set, ntrain_data, replace=False)
	for text in train_data_neg:
		if text not in GO2text[GO]:
			train_data.append((text,0))
	lr = LogisticRegression()
	text_clf = TextClassify()
	text_clf.train(lr,train_data)
	with open(GO_term_file, 'wb') as output:
		pickle.dump(text_clf, output, pickle.HIGHEST_PROTOCOL)



	text_clf = pickle.load(open(GO_term_file, "rb" ))
	GO2keyword[GO] = text_clf.get_keyword(nkeyword)

	nsent = 0
	sent2prob = {}
	#print self.pubmed_dir
	for pid in range(nchunk):
		fin = open(pubmed_dir+'pmid2meta_autophrase.chunk'+str(pid))
		for line in fin:
			text_l = [line.lower().translate(None,',?!:()=%>/[]').strip().strip('.')]
			for text in text_l:
				pset = parse_sentence(GO2keyword[GO],text,max_phrase_length = 5)
				if len(pset) > 0 and GO not in text:
					lab = text_clf.predict(text)
					prob = text_clf.predict_prob(text)
					if lab==1 and prob > 0.8:
												sent2prob[text] = prob
						#fout.write(str(prob)+'\t'+text+'\n')
						nsent += 1
		fin.close()
	fout = open(text_file,'w')
		for text in sent2prob:
			fout.write(text+'\t'+str(prob)+'\n')
	fout.close()

def call_merge_preprocess(i,npid,GO_term_set,GO_term_dir,GO2text,negative_set):
	for ct,GO in enumerate(GO_term_set):
		predict_GO_term(GO,GO_term_dir,GO2text,negative_set)
		if ct%10==0:
			print 'finished',ct*1.0/len(GO_term_set)

nchunk = 400
pubmed_dir = '/data/cellardata/users/majianzhu/wangsheng/Sheng_repo/pubmed/'

GO_term_dir = '/cellar/users/majianzhu/Data/wangsheng/Sheng_repo/data/Pubmed/GO_sentences/'
if not os.path.exists(GO_term_dir):
	os.makedirs(GO_term_dir)

GO_term_file = '/data/cellardata/users/majianzhu/wangsheng/Sheng_repo/Sheng_repo/data/NLP_Dictionary/go_term.txt'
GO_term = set()

fin = open(GO_term_file)
for line in fin:
	GO_term.add(line.lower().strip())
fin.close()

#GO_term.add('dna repair')

GO_term_ct = {}
GO2text = {}

sent_collection = []
negative_set = []
negative_thres = 0.999
for pid in range(nchunk):
	fin = open(pubmed_dir+'pmid2meta_autophrase.chunk'+str(pid))
	for line in fin:
		text_l = [line.lower().translate(None,',?!:()=%>/[]').strip().strip('.')]
		for text in text_l:
			pset = parse_sentence(GO_term,text,max_phrase_length = 5)
			for p in pset:
				if p not in GO2text:
					GO2text[p] = {}
				GO2text[p][text] = 1
			r = random.uniform(0, 1)
			if r>negative_thres:
				negative_set.append(text)
	fin.close()
	print pid,len(negative_set)
negative_set = np.array(negative_set)

npid = 32
p2chunk = {}
for i in range(npid):
	p2chunk[i] = set()
for ct,GO in enumerate(GO_term):
	if GO not in GO2text:
		continue
	p2chunk[ct%npid].add(GO)

jobs = []
for i in range(npid):
	p = multiprocessing.Process(target=call_merge_preprocess, args=(i,npid,p2chunk[i],GO_term_dir,GO2text,negative_set))
	jobs.append(p)
	p.start()
for proc in jobs:
	proc.join()


