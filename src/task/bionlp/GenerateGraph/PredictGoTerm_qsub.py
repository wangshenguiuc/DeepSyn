import multiprocessing
import sys
import random
import os
import pickle
import operator
import collections
import numpy as np
import time
sys.path.append(repo_dir)
os.chdir(repo_dir)

from src.datasets.WordNet import parse_sentence
from src.models.text_classification.TextClassify import TextClassify
from sklearn.linear_model import LogisticRegression
from PredictGoTerm_utils import *

proc_id = int(sys.argv[1])
totalpid = int(sys.argv[2])
nchunk = 400
mode = 3#0: extract abs, 1: train_model, 2:generate sent
server = 'sherlock'#'grenache''sherlock'
if server == 'timan107':
	pubmed_dir = '/srv/local/work/swang141/PatientSetAnnotation/Sheng/data/pubmed/'
	GO_data_dir = '/srv/local/work/swang141/Sheng_repo/data/Pubmed/'
	GO_term_file = '/srv/local/work/swang141/Sheng_repo/data/NLP_Dictionary/go_term.txt'
	repo_dir = '/srv/local/work/swang141/Sheng_repo/'
elif server == 'grenache':
	pubmed_dir = '/data/cellardata/users/majianzhu/wangsheng/Sheng_repo/pubmed/'
	GO_data_dir = '/cellar/users/majianzhu/Data/wangsheng/Sheng_repo/data/Pubmed/'
	GO_term_file = '/data/cellardata/users/majianzhu/wangsheng/Sheng_repo/Sheng_repo/data/NLP_Dictionary/go_term.txt'
	repo_dir = '/data/cellardata/users/majianzhu/wangsheng/Sheng_repo/Sheng_repo/'
else:
	pubmed_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/Pubmed/pubmed/'
	GO_data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/Pubmed/'
	GO_term_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/NLP_Dictionary/go_term.txt'
	repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'


GO_text_dir = GO_data_dir+'GO_abstracts_test/'
GO_pred_dir = GO_data_dir + 'GO_models_L2/'
GO_sent_dir = GO_data_dir + 'GO_sentences_L2/'
GO_sent_merge_dir = GO_data_dir + 'GO_sentences_L2_all/'

if not os.path.exists(GO_sent_merge_dir):
	os.makedirs(GO_sent_merge_dir)
if not os.path.exists(GO_sent_dir):
	os.makedirs(GO_sent_dir)
if not os.path.exists(GO_pred_dir):
	os.makedirs(GO_pred_dir)
if not os.path.exists(GO_text_dir):
	os.makedirs(GO_text_dir)


for i in range(totalpid):
	GO_sent_dir_pid = GO_sent_dir +str(i)+'/'
	if not os.path.exists(GO_sent_dir_pid):
		os.makedirs(GO_sent_dir_pid)


pred_GO_term = set()
abs_GO_term = set()
sent_GO_term = set()
fin = open(GO_term_file)
ct = 0
for line in fin:
	GO = line.lower().strip()
	text_file = GO_text_dir + GO.replace('/','_').replace(' ','_')+'.pkl'
	pred_file = GO_pred_dir + GO.replace('/','_').replace(' ','_')+'.pkl'
	sent_file = GO_sent_dir + GO.replace('/','_').replace(' ','_')+'.txt'
	if mode == 1 and not os.path.isfile(pred_file) and os.path.isfile(text_file):
		ct+=1
		if ct%totalpid == proc_id:
			pred_GO_term.add(GO)
	if mode == 0 and not os.path.isfile(text_file):
		ct+=1
		if ct%totalpid == proc_id:
			abs_GO_term.add(GO)
	if mode == 2 and not os.path.isfile(sent_file) and os.path.isfile(pred_file):
		ct+=1
		if ct%totalpid == proc_id:
			sent_GO_term.add(GO)
fin.close()
GO_term_ct = {}
GO2text = {}
print sent_GO_term

if mode==1 or mode == 0:
	sent_collection = []
	negative_set = []
	negative_thres = 0.99
	if len(abs_GO_term) > 0:
		for pid in range(nchunk):
			fin = open(pubmed_dir+'pmid2meta_autophrase.chunk'+str(pid))
			for line in fin:
				text_l = [line.lower().translate(None,',?!:()=%>/[]').strip().strip('.')]
				for text in text_l:
					pset = parse_sentence(abs_GO_term,text,max_phrase_length = 5)
					for p in pset:
						if p not in GO2text:
							GO2text[p] = {}
						GO2text[p][text] = 1
			fin.close()
			print pid,len(negative_set)
			sys.stdout.flush()


	for ct, GO in enumerate(GO2text):
		GO2text_file = GO_text_dir+GO.replace('/','_').replace(' ','_')+'.pkl'
		if os.path.isfile(GO2text_file):
			continue
		with open(GO2text_file, 'wb') as output:
			pickle.dump(GO2text[GO], output, pickle.HIGHEST_PROTOCOL)

	for ct, GO in enumerate(pred_GO_term):
		text_file = GO_text_dir + GO.replace('/','_').replace(' ','_')+'.pkl'
		if not os.path.isfile(text_file):
			continue
		GO2text = pickle.load(open(text_file, "rb" ))
		for text in GO2text:
			r = random.uniform(0, 1)
			if r>negative_thres:
				negative_set.append(text.replace(GO,''))
		print ct, len(pred_GO_term), len(negative_set)

	negative_set = np.array(negative_set)
	print 'read data finished number of negatives',len(negative_set)
	sys.stdout.flush()
	call_merge_preprocess(pred_GO_term,GO_text_dir,GO_pred_dir,GO_sent_dir,negative_set)
elif mode==2:
	pred_GO_term = set()
	abs_GO_term = set()
	sent_GO_term = set()
	fin = open(GO_term_file)
	ct = 0
	for line in fin:
		GO = line.lower().strip()
		sent_file = GO_sent_dir + str(proc_id)+'/'+ GO.replace('/','_').replace(' ','_')+'.txt'
		if os.path.isfile(sent_file):
			continue
		pred_file = GO_pred_dir + GO.replace('/','_').replace(' ','_')+'.pkl'
		if not os.path.isfile(pred_file):
			continue
		sent_GO_term.add(GO)
		ct += 1
	print 'ngo',len(sent_GO_term)
	sys.stdout.flush()
	nchunk = 400
	p2chunk = {}
	for i in range(nchunk):
		pid = i%totalpid
		if pid not in p2chunk:
			p2chunk[pid] = set()
		p2chunk[pid].add(i)
	call_predict_preprocess(proc_id,totalpid,p2chunk[proc_id-1],sent_GO_term,GO_pred_dir,GO_sent_dir)
elif mode==3:
	sent_GO_term = set()
	fin = open(GO_term_file)
	ct = 0
	for line in fin:
		GO = line.lower().strip()
		sent_GO_term.add(GO)
		ct += 1
	fin.close()
	for ct,GO in enumerate(sent_GO_term):
		print GO,ct
		sys.stdout.flush()
		new_sent_file = GO_sent_merge_dir +  GO.replace('/','_').replace(' ','_')+'.txt'
		fout = open(new_sent_file,'w')
		for i in range(totalpid):
			print i
			sys.stdout.flush()
			sent_file = GO_sent_dir + str(i)+'/'+ GO.replace('/','_').replace(' ','_')+'.txt'
			if not os.path.isfile(sent_file):
				print i,GO,'not found'
				sys.stdout.flush()
				continue
			fin = open(sent_file)
			for line in fin:
				fout.write(line.strip()+'\n')
			fin.close()
		fout.close()
else:
	pass
