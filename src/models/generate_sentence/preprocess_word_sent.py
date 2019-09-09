import multiprocessing
import sys
import os
import subprocess
import collections
from src.datasets.WordNet import WordNet,parse_sentence
from src.utils.nlp import parse_word_net
#from src.models.generate_sentence import run_preprocess_word_sent

class ReadPreProcessWordSent():
	def __init__(self,CACHE_DIR,DATA_DIR):
		self.CACHE_DIR = CACHE_DIR
		self.DATA_DIR = DATA_DIR
		self.predict_dir = self.DATA_DIR+'data/Pubmed/GO_sentences_L2_all_withpmid/'
		self.word_dir = self.DATA_DIR+'data/Pubmed/word_sentences/new_words/'
		self.sent_dir = self.DATA_DIR+'data/Pubmed/word_sentences/sents/'
		self.pubmed_dir = self.DATA_DIR+ 'data/Pubmed/pubmed/'
	def get_word_path(self,word,suffix):
		suffix = str(suffix)
		if len(word)<=2:
			word += '~'
		step = 2
		dir = self.word_dir + ''
		nchar = len(word)
		word = word.replace(' ','_')
		for i in range(0,nchar,step):
			dir += word[i:min(i+step,nchar)] + '/'
		if not os.path.exists(dir):
			os.makedirs(dir)
		file = dir + word + suffix
		return file,dir

	def get_sentence_word(self,word,generate_miss_sentence=True):
		word_path = self.get_word_path(word,'')[0]
		pos2sent = {}
		if os.path.isfile(word_path):
			fin = open(word_path)
			for line in fin:
				cid,line_ct,sent_ct,sent = line.strip().split('\t')
				pos2sent[(cid,line_ct,sent_ct)] = sent
			fin.close()
		if len(pos2sent) == 0 and generate_miss_sentence:
			print 'preprocess word',word
			write_word_sents_to_file(set([word]),DATA_DIR=self.DATA_DIR,CACHE_DIR=self.CACHE_DIR,npid=2)
			word_path = self.get_word_path(word,'')[0]
			pos2sent = {}
			fin = open(word_path)
			for line in fin:
				cid,line_ct,sent_ct,sent = line.strip().split('\t')
				pos2sent[(cid,line_ct,sent_ct)] = sent
			fin.close()
		return pos2sent

	def get_word_pair_path(self,w1,w2):
		w2 = w2.replace(' ','_')
		w1 = w1.replace(' ','_')
		if w1<w2:
			w1,w2 = w2,w1
		word = w1 + '~' + w2
		dir = self.get_word_path(word,'')[1]
		if not os.path.exists(dir):
			os.makedirs(dir)
		if os.path.isdir(dir + word):
			os.rmdir(dir + word)
		return dir,word

	def sentence_edge_is_in_cache(self,w1,w2):
		dir, word = self.get_word_pair_path(w1,w2)
		file = dir + word
		sent_l = []
		if os.path.isfile(file):
			fin = open(file)
			for line in fin:
				sent_l.append(line.strip())
			return True,sent_l
		return False,sent_l


	def get_sentence_edge(self,w1,w2,use_cache=False):

		if use_cache:
			is_in_cache,sent_l = self.sentence_edge_is_in_cache(w1,w2)
			if is_in_cache:
				return sent_l
		pos2sent1 = self.get_sentence_word(w1,generate_miss_sentence=False)
		pos2sent2 = self.get_sentence_word(w2,generate_miss_sentence=False)
		sent_l = []
		title_l = []
		pmid_l = []
		dir, word = self.get_word_pair_path(w1,w2)
		file = dir + word
		fout = open(file,'w')
		
		for p in pos2sent1:
			if p in pos2sent2:
				sent_l.append(pos2sent1[p])
				title,pmid = self.get_title_sentence(p)
				title_l.append(title)
				pmid_l.append(pmid)
				sent_l.append(pos2sent1[p].strip())
				fout.write(pos2sent1[p].strip()+'\n')
		fout.close()
		return sent_l,title_l,pmid_l
	
	def get_title_sentence(self,p):
		#pos2sent[(cid,line_ct,sent_ct)]
		cid,line_ct_best,sent_ct = p
		line_ct_best = int(line_ct_best)
		file_name = self.pubmed_dir + '/pmid2meta_autophrase.chunk' + str(cid)
		fin = open(file_name,'r')
		for line_ct,line in enumerate(fin):
			if line_ct == line_ct_best:
				text = line.lower().translate(None, ',:()=%>/[]')
				sent_l = text.strip().split('.')
				title = sent_l[0]
				break
		fin.close()
		file_name = self.pubmed_dir + '/pmid.chunk'+str(cid)+'step2'
		fin = open(file_name,'r')
		for line_ct,line in enumerate(fin):
			if line_ct == line_ct_best:
				pmid = line.strip()
				break
		fin.close()
		if line_ct_best > line_ct:
			print file_name, line_ct, line_ct_best
		return title,pmid
	
	def get_inferred_sentence_edge(self,w1,w2):
		file1 = self.predict_dir + w1.replace(' ','_')+'.txt'
		file2 = self.predict_dir + w2.replace(' ','_')+'.txt'
		sent_l = []
		#print file1,file2,os.path.isfile(file1),os.path.isfile(file2)
		if os.path.isfile(file1):
			#print file1
			fin = open(file1)
			for line in fin:
				w = line.strip().split('\t')[0]
				if w2 in w:
					sent_l.append(w)
			fin.close()
		if os.path.isfile(file2):
			#print file2
			fin = open(file2)
			for line in fin:
				w = line.strip().split('\t')[0]
				if w1 in w:
					sent_l.append(w)
			fin.close()
		if len(sent_l) == 0:
			sys.exit(w1+'\t'+w2+'\tnot found\n')
		return sent_l

class PreprocessWordSent():
	def __init__(self,word_set,max_len,DATA_DIR,CACHE_DIR):
		self.word_set = word_set
		self.max_len = max_len
		self.DATA_DIR = DATA_DIR
		self.CACHE_DIR = CACHE_DIR
		self.pubmed_dir = self.DATA_DIR+ 'data/Pubmed/pubmed/'
		self.pos2sent = {}
		self.word2pos = collections.defaultdict(set)
		self.sent2ct = {}
		self.ReadWordSent_obj = ReadPreProcessWordSent(DATA_DIR=self.DATA_DIR,CACHE_DIR=self.CACHE_DIR)

	def merge(self,pid_l,cur_p=-1):
		nword = len(self.word_set)
		for ct,w in enumerate(self.word_set):
			word_path = self.ReadWordSent_obj.get_word_path(w,'')[0]
			if os.path.isfile(word_path):
				print w,'exist'
				#continue
			fout = open(word_path,'w')
			for pid in pid_l:
				word_path = self.ReadWordSent_obj.get_word_path(w,pid)[0]
				if not os.path.isfile(word_path):
					continue
				fin = open(word_path)
				for line in fin:
					fout.write(line.strip()+'\n')
				fin.close()
				os.remove(word_path)
			fout.close()
			if ct%(int(nword/10)+1)==0 and cur_p==0:
				print 'merge all files finished',ct*1.0/nword,nword
				sys.stdout.flush()

	def write2file(self,pid):
		nword = len(self.word2pos)
		for ct,p in enumerate(self.word2pos):
			word_path = self.ReadWordSent_obj.get_word_path(p,pid)[0]
			fout = open(word_path,'w')
			for cid,line_ct,sent_ct in self.word2pos[p]:
				sent = self.pos2sent[(cid,line_ct,sent_ct)]
				fout.write(str(cid)+'\t'+str(line_ct)+'\t'+str(sent_ct)+'\t'+sent+'\n')
			fout.close()
			if ct%(int(nword/10)+1)==0 and pid==0:
				print 'write to file finished',ct*1.0/nword
				sys.stdout.flush()


	def process(self,cid):
		ct = 0
		edge_ct = {}
		edge_ref = {}
		wct = {}
		file_name = self.pubmed_dir + '/pmid2meta_autophrase.chunk' + str(cid)
		fin = open(file_name,'r')
		for line_ct,line in enumerate(fin):
			text = line.lower().translate(None, ',:()=%>/[]')
			sent_l = text.strip().split('.')
			if line_ct%10000==0:
				#print 'write lines finished',line_ct
				sys.stdout.flush()
			for sent_ct,sent in enumerate(sent_l):
				pset = parse_sentence(self.word_set,sent,self.max_len)
				for p in pset:
					self.word2pos[p].add((cid,line_ct,sent_ct))
					self.pos2sent[(cid,line_ct,sent_ct)] = sent


def call_preprocess(pid,chunk_set,word_set,max_len,DATA_DIR,CACHE_DIR):
	Pre_obj = PreprocessWordSent(word_set,max_len,DATA_DIR,CACHE_DIR)
	for cid in chunk_set:
		print cid,'finished'
		sys.stdout.flush()
		Pre_obj.process(cid)
	if pid==0:
		print pid,'finished reading'
	Pre_obj.write2file(pid)


def call_merge_preprocess(pid,npid,word_set,max_len,DATA_DIR,CACHE_DIR):
	Pre_obj = PreprocessWordSent(word_set,max_len,DATA_DIR,CACHE_DIR)
	Pre_obj.merge(range(0,npid),pid)


def write_word_sents_to_file(word_set,DATA_DIR,CACHE_DIR,npid=64,nchunk=400,pid=-1,max_len=4):
	max_len = max_len
	p2chunk = {}

	for i in range(npid):
		p2chunk[i] = []
	for c in range(nchunk):
		p2chunk[c%npid].append(c)
	if pid == -1:
		manager = multiprocessing.Manager()
		jobs = []
		for i in range(npid):
			p = multiprocessing.Process(target=call_preprocess, args=(i,p2chunk[i],word_set,max_len,DATA_DIR,CACHE_DIR))
			jobs.append(p)
			p.start()
		for proc in jobs:
			proc.join()

		p2chunk = {}
		for i in range(npid):
			p2chunk[i] = set()
		for c,w in enumerate(word_set):
			p2chunk[c%npid].add(w)

		jobs = []
		for i in range(npid):
			p = multiprocessing.Process(target=call_merge_preprocess, args=(i,npid,p2chunk[i],max_len,DATA_DIR,CACHE_DIR))
			jobs.append(p)
			p.start()
		for proc in jobs:
			proc.join()
	else:
		call_preprocess(pid,p2chunk[pid],word_set,max_len,DATA_DIR,CACHE_DIR)
		call_merge_preprocess(pid,npid,word_set,max_len,DATA_DIR,CACHE_DIR)
