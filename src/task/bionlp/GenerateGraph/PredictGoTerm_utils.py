
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

def train_model(GO, GO2text, text_file, pred_file, log_file, nfold = 3):
	print GO,'train'
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
	train_data_neg = generate_train_model(negative_set, ntrain_data)
	print GO,len(train_data),'positive samples'
	sys.stdout.flush()
	for text in train_data_neg:
		if text not in GO2text:
			train_data.append((text,0))
	print GO,len(train_data),'all samples'
	sys.stdout.flush()
	lr = LogisticRegression()#penalty='l1',C=0.1
	text_clf = TextClassify()

	train_data = np.array(train_data)
	cv_data = train_data
	np.random.shuffle(cv_data)
	ratio = 1 - 1.0 / nfold
	cv_train_data = cv_data[:np.int(ratio*len(cv_data))]
	cv_test_data = cv_data[np.int(ratio*len(cv_data)):]
	fout = open(log_file,'w')
	has_pos = False
	for t in cv_train_data:
		if t[1] == '1':
			has_pos = True
			break
	if has_pos:
		text_clf.train(lr,cv_train_data)
		test_acc = text_clf.evaluate(cv_test_data)
		train_acc = text_clf.evaluate(cv_train_data)
		fout.write(str(test_acc)+'\t'+str(train_acc)+'\t'+str(text_clf.nword)+'\n')
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


def call_merge_preprocess(GO_term_set,GO_text_dir,GO_pred_dir,GO_sent_dir,negative_set):
	for ct, GO in enumerate(GO_term_set):
		text_file = GO_text_dir + GO.replace('/','_').replace(' ','_')+'.pkl'
		if not os.path.isfile(text_file):
			continue
		print GO
		sys.stdout.flush()
		GO2text = pickle.load(open(text_file, "rb" ))
		pred_file = GO_pred_dir + GO.replace('/','_').replace(' ','_')+'.pkl'
		log_file = GO_pred_dir + GO.replace('/','_').replace(' ','_')+'.log'
		if not os.path.isfile(pred_file):
			print GO,len(GO2text)
			sys.stdout.flush()
			#try:
			train_model(GO,GO2text,text_file,pred_file,log_file)
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
