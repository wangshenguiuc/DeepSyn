import multiprocessing
import sys
import random
import os
import pickle
import operator
import collections
import numpy as np
import time
import psutil
import GPUtil



def write_GO_to_file(pid, GO_sent_dir, GO2text):
	cal_feat_time = 0.
	pickle_time = 0.
	cal_prob_time = 0.
	max_abs = 50
	for ct,GO in enumerate(GO2text):
		sent2prob = {}
		sent_file = GO_sent_dir + str(pid)+'/'+ GO.replace('/','_').replace(' ','_')+'.txt'

		pred_file =	GO_pred_dir + GO.replace('/','_').replace(' ','_')+'.pkl'
		start = time.time()
		try:
			text_clf = pickle.load(open(pred_file, "rb" ))
		except:
			print 'unable to open',GO
			continue
		end = time.time()
		pickle_time += end - start
		text2prob = {}
		new_GO_l = sorted(GO2text[GO].items(), key = lambda item:item[1])
		ct = 0
		for text,sc in new_GO_l:
			'''
			start = time.time()
			feat = [text_clf.get_test_feature(text)]
			end = time.time()
			cal_feat_time += end - start
			start = time.time()
			#prob = text_clf.predict_prob(text)
			#prob = text_clf.predict_prob_feat(feat)
			feat_pos = np.nonzero(text_clf.clf.coef_)[1]
			sc = text_clf.clf.intercept_[0]
			for f in feat_pos:
				sc += text_clf.clf.coef_[0][f] * feat[0][f]
			prob = 1 / (1 + np.exp(-1*sc))
			prob1 = text_clf.predict_prob_feat(feat)
			print prob,prob1
			end = time.time()
			cal_prob_time += end - start
			'''
			prob = text_clf.predict_prob(text)
			if prob > 0.9:
				text2prob[text] = prob*-1
				if len(text2prob) > max_abs:
					break
			ct += 1
			if ct > 10000:
				break
				#fout.write(text+'\t'+str(prob)+'\n')
		fout = open(sent_file,'w')
		text2prob_sort = sorted(text2prob.items(), key = lambda item:item[1])
		noutput = min(max_abs, len(text2prob_sort))
		for i in range(noutput):
			text = text2prob_sort[i][0]
			fout.write(text +'\t' + str(text2prob_sort[i][1])+'\t'+str(GO2text[GO][text])+'\n')
		fout.close()
		if ct%1==0:
			print GO,
			sys.stdout.flush()


def predict_GO_term(proc_id, keyword2GO,p2chunk,GO_sent_dir,GO_pred_dir):
	keyword_set = set(keyword2GO.keys())
	print len(keyword_set)
	parse_time = 0.
	append_time = 0.
	nsent = 0
	GO2text = {}
	#print self.pubmed_dir
	for pid in p2chunk:
		fin = open(pubmed_dir+'pmid2meta_autophrase.chunk'+str(pid))
		for ct,line in enumerate(fin):
			text_l = [line.lower().translate(None,',?!:()=%>/[]').strip().strip('.')]
			for text in text_l:
				start = time.time()
				pset = parse_sentence(keyword_set,text,max_phrase_length = 1)
				end = time.time()
				parse_time += end - start
				start = time.time()
				cur_GO = {}
				nword = len(text.split(' '))
				for p in pset:
					for GO in keyword2GO[p]:
						if GO in text:
							continue
						cur_GO[GO] = cur_GO.get(GO,0) + 1
				for GO in cur_GO:
					if GO not in GO2text:
						GO2text[GO] = {}
					GO2text[GO][text] = cur_GO[GO] * 1.0 / nword *-1
					#fout.write(str(prob)+'\t'+text+'\n')
					nsent += 1
				end = time.time()
				append_time += end - start
			if ct%5000==0:
				print proc_id, pid, 'read finished',ct/40000.,nsent,append_time,parse_time
				sys.stdout.flush()
		print proc_id, pid,'start to write file',parse_time,append_time,nsent
		sys.stdout.flush()
		write_GO_to_file(proc_id,GO_sent_dir,GO2text)
		GO2text = {}
		print proc_id, pid,'end to write file'
		sys.stdout.flush()
		fin.close()

def generate_train_model(list, n):
	N = len(list)
	ind = np.random.choice(N, n)
	new_list = []
	for i in ind:
		new_list.append(list[i])
	return new_list
	#for i in range(n):
	#	yield np.random.choice(list, 1)

def train_model(GO, GO2text, text_file, pred_file, log_file, auc_file, nfold = 3):

	sys.stdout.flush()
	GO2keyword = {}
	nmax_train_data = 3000
	nmin_train_data = 3000

	train_data = []
	if len(GO2text) > nmax_train_data:
		select_sent = generate_train_model(GO2text.keys(), nmax_train_data)
		for t in select_sent:
			train_data.append((t.replace(GO,''),1))
	else:
		for t in GO2text:
			train_data.append((t.replace(GO,''),1))
	sys.stdout.flush()
	ntrain_data = min(nmin_train_data, len(negative_set))
	#print ntrain_data
	train_data_neg = generate_train_model(negative_set, ntrain_data)
	#print GO,len(train_data),'positive samples',len(train_data_neg),'negative sample'
	sys.stdout.flush()
	for text in train_data_neg:
		if text not in GO2text:
			train_data.append((text,0))
	npos = 0
	nneg = 0
	for i in train_data:
		f,l = i
		if l==1:
			npos+=1
		elif l==0:
			nneg += 1
		else:
			sys.exit('wrong label')
	if npos < 100:
		return
	#print npos,nneg
	print GO,len(train_data),'all samples'
	sys.stdout.flush()
	cv_data = train_data
	np.random.shuffle(cv_data)
	classify_d = {}
	classify_d['lr'] = LogisticRegression()
	#classify_d['rnn'] = DeepText(working_dir = repo_dir, DeepModel = RNN,use_glove=True)
	#classify_d['lstm'] = DeepText(working_dir = repo_dir, DeepModel = LSTM,use_glove=True)
	classify_d['cnn_glove'] = DeepText(working_dir = repo_dir, DeepModel = CNN,use_glove=True)
	#classify_d['cnn'] = DeepText(working_dir = repo_dir, DeepModel = CNN)
	#lr = TextCNN(working_dir = repo_dir)#penalty='l1',C=0.1
	text_clf = TextClassify()

	ratio = 1 - 1.0 / nfold
	cv_train_data = cv_data[:np.int(ratio*len(cv_data))]
	cv_test_data = cv_data[np.int(ratio*len(cv_data)):]
	fout = open(log_file,'w')
	has_pos = False
	for t in cv_train_data:
		if t[1] == 1:
			has_pos = True
			break
	if has_pos:
		for clf in classify_d:
			lr = classify_d[clf]
			if clf=='lr':
				feat_type = 'vector'
			else:
				feat_type = 'word'
			start_time = time.time()

			text_clf.train(lr,cv_train_data,feat_type = feat_type)
			# your code
			elapsed_time = time.time() - start_time
			#print elapsed_time
			start_time = time.time()
			test_acc = text_clf.evaluate(cv_test_data)
			elapsed_time = time.time() - start_time
			#print elapsed_time
			start_time = time.time()
			train_acc = text_clf.evaluate(cv_train_data)
			elapsed_time = time.time() - start_time
			#print elapsed_time
			fauc = open(auc_file,'w')
			fauc.write(GO+'\t'+str(test_acc)+'\t'+str(train_acc)+'\n')
			print GO, clf, test_acc, train_acc,npos,nneg
			sys.stdout.flush()
			fauc.close()
			fout.write(str(test_acc)+'\t'+str(train_acc)+'\t'+str(text_clf.nword)+'\n')
		'''
		para = text_clf.clf.coef_[0]*-1
		word_ind = np.argsort(para)[:20]
		for i in range(20):
			wi = word_ind[i]
			fout.write(str(i)+'\t'+str(text_clf.i2w[wi])+'\t'+str(text_clf.clf.coef_[0][wi])+'\n')

	#lr = LogisticRegression(penalty='l1',C=1) #
	#print test_acc, train_acc,text_clf.nword
	text_clf.train(lr,train_data)
	para = text_clf.clf.coef_[0]*-1
	word_ind = np.argsort(para)[:20]
	for i in range(20):
		wi = word_ind[i]
		fout.write(str(i)+'\t'+str(text_clf.i2w[wi])+'\t'+str(text_clf.clf.coef_[0][wi])+'\n')
	sys.stdout.flush()
	fout.close()
	with open(pred_file, 'wb') as output:
		pickle.dump(text_clf, output, pickle.HIGHEST_PROTOCOL)
	'''

def call_merge_preprocess(GO_term_set,GO_text_dir,GO_pred_dir,GO_sent_dir,negative_set):
	for ct, GO in enumerate(GO_term_set):
		text_file = GO_text_dir + GO.replace('/','_').replace(' ','_')+'.pkl'
		if not os.path.isfile(text_file):
			continue
		#print GO
		sys.stdout.flush()
		GO2text = pickle.load(open(text_file, "rb" ))
		pred_file = GO_pred_dir + GO.replace('/','_').replace(' ','_')+'.pkl'
		log_file = GO_pred_dir + GO.replace('/','_').replace(' ','_')+'.log'
		auc_file = GO_pred_dir + 'auc.txt'
		if not os.path.isfile(pred_file):
			#print GO,len(GO2text)
			sys.stdout.flush()
			#try:
			train_model(GO,GO2text,text_file,pred_file,log_file,auc_file)
			#except Exception, e:
			#	continue
		if not os.path.isfile(pred_file):
			continue
		text_clf = pickle.load(open(pred_file, "rb" ))

def call_predict_preprocess(i,npid,p2chunk,GO_term_set,GO_pred_dir,GO_sent_dir):
	start = time.time()
	keyword2GO = {}
	nkeyword = 20
	ct = 0
	print len(GO_term_set)
	sys.stdout.flush()
	for ct, GO in enumerate(GO_term_set):
		if ct%100==0:
			end = time.time()
			print 'len keyword',ct,end-start,len(GO_term_set),len(keyword2GO)
			sys.stdout.flush()
		pred_file = GO_pred_dir + GO.replace('/','_').replace(' ','_')+'.pkl'
		keyword_file = GO_pred_dir + GO.replace('/','_').replace(' ','_')+'.keyword.pkl'
		if not os.path.isfile(pred_file):
			continue

		if os.path.isfile(keyword_file):
			key_word = pickle.load(open(keyword_file, "rb" ))
		else:
			text_clf = pickle.load(open(pred_file, "rb" ))
			key_word = text_clf.get_keyword(nkeyword)
			with open(keyword_file, 'wb') as output:
				pickle.dump(key_word, output, pickle.HIGHEST_PROTOCOL)
		#print GO,key_word
		sys.stdout.flush()
		for w in key_word:
			if w not in keyword2GO:
				keyword2GO[w] = []
			keyword2GO[w].append(GO)

	end = time.time()
	print 'len keyword',ct,end-start
	sys.stdout.flush()
	predict_GO_term(i, keyword2GO,p2chunk,GO_sent_dir,GO_pred_dir)

proc_id = int(sys.argv[1])
totalpid = int(sys.argv[2])
mode = int(sys.argv[3])
nchunk = 400
#mode = 3#0: extract abs, 1: train_model, 2:generate sent, 4: merge sent
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


sys.path.append(repo_dir)
os.chdir(repo_dir)


from src.models.text_classification.DeepText import DeepText
from src.datasets.WordNet import parse_sentence
from src.models.text_classification.LSTM import LSTM as LSTM
from src.models.text_classification.RNN import RNN as RNN
from src.models.text_classification.CNN import CNN
from src.models.text_classification.TextClassify import TextClassify
from sklearn.linear_model import LogisticRegression


GO_text_dir = GO_data_dir+'GO_abstracts_test/'
#GO_pred_dir = GO_data_dir + 'GO_models_L2/'
GO_pred_dir = GO_data_dir + 'GO_models_CNN/'
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
print len(pred_GO_term)
np.random.seed(0)
random.seed(0)
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
		#print ct, len(pred_GO_term), len(negative_set)

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
			os.remove(sent_file)
		fout.close()
else:
	sys.exit('wrong mode')
