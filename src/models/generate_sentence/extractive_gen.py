from nltk.stem.porter import *
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import os
import subprocess
import operator
import collections
import sys
import operator

from src.models.generate_sentence.preprocess_word_sent import ReadPreProcessWordSent
from src.utils.nlp import parse_word_net

class ExtractGenSent():

	def get_all_sentences(self,w1,w2):
		sent_collection = []
		#print self.pubmed_dir
		for pid in range(self.nchunk):
			fin = open(self.pubmed_dir+'pmid2meta_autophrase.chunk'+str(pid))
			for line in fin:
				text = line.lower().translate(None,',?!:()=%>/[]').strip().strip('.').split('. ')
				sent_l = text
				for sent in sent_l:
					if w1 in sent and w2 in sent:
						sent_collection.append(sent.replace('.','')+' .')
			fin.close()
			if pid%50==0:
				#print 'read',pid,len(sent_collection)
				pass
		return sent_collection

	def find_sent_pubmed(self,w1,w2):
		sent_l,title_l,pmid_l = self.ReadWordSent_obj.get_sentence_edge(w1,w2)
		return sent_l,title_l,pmid_l

	def find_sent_infer(self,w1,w2):
		#sent_l = get_inferred_sentence_edge(w1,w1,predict_dir=self.infer_sentence_dir)
		if os.path.isfile(self.infer_sentence_dir+w1.replace(' ','_')+'.txt.withpmid'):
			go_term = w1
			word = w2
			file = self.infer_sentence_dir+w1.replace(' ','_')+'.txt.withpmid'
		elif os.path.isfile(self.infer_sentence_dir+w2.replace(' ','_')+'.txt.withpmid'):
			go_term = w2
			word = w1
			file = self.infer_sentence_dir+w2.replace(' ','_')+'.txt.withpmid'
		else:
			print 'wrong GO term '+w1 +'#'+w2,self.infer_sentence_dir+w2.replace(' ','_')+'.txt.withpmid'
			return '','',''
			#sys.exit('wrong GO term'+w1+'#'+w2+'\n')
		sent_l = []
		title_l = []
		pmid_l = []
		fin = open(file)
		for line in fin:
			text = line.strip().split('\t')[0]
			pmid = line.strip().split('\t')[-1]
			if pmid == '-1':
				continue
			text = text.lower().translate(None,',?!:()=%>/[]').strip().strip('.').split('. ')
			title = text[0]
			
			for sent in text:
				#ww = sent.split(' ')
				if word in sent:
					sent_l.append(sent.replace('.','')+' .')
					title_l.append(title)
					pmid_l.append(pmid)
		fin.close()
		return sent_l, title_l, pmid_l

	def __init__(self,DATA_DIR='/oak/stanford/groups/rbaltman/swang91/Sheng_repo/',
	CACHE_DIR = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/',
	nlp_tool_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/tools/stanford_parser_java/stanford-parser-full-2017-06-09/',nchunk = 400,topk_key_word=2, topk_sent = 5):
		self.topk_key_word = topk_key_word
		self.DATA_DIR = DATA_DIR
		self.CACHE_DIR = CACHE_DIR
		self.nchunk = nchunk
		self.nlp_tool_dir = nlp_tool_dir
		self.infer_sentence_dir = self.DATA_DIR + 'data/Pubmed/GO_sentences_L2_all_withpmid/'
		self.pubmed_dir = self.DATA_DIR + 'data/Pubmed/pubmed/'
		self.word_dir = self.DATA_DIR + 'data/Pubmed/word_sentences/new_words/'
		self.topk_sent = topk_sent
		self.ReadWordSent_obj = ReadPreProcessWordSent(DATA_DIR=self.DATA_DIR,CACHE_DIR=self.CACHE_DIR)

	def get_graph_sentence(self,graph_edge_list,graph_file_prefix, call_sentence=True):
		nedge = len(graph_edge_list)
		new_graph_obj = []
		pm_title = ''
		for e in graph_edge_list:
			e1,e2,w,type_list = e

			if e1 == '@super target' or e2 == '@super target':
				continue
			if not call_sentence:
				new_graph_obj.append([e1,e2,type_list,'','',''])
				continue

			if 'pubmed' in type_list:
				sent_l, title_l, pmid_l = self.find_sent_pubmed(e2,e1)[:3]
			elif 'infer' in type_list:
				sent_l, title_l, pmid_l = self.find_sent_infer(e2,e1)
			else:
				pmid_l = ['']
				title_l = ['']
				sent_l = ['from database']
			#pm_title = 'Characterizing Cancer Drug Response and Biological Correlates: A Geometric Network Approach'
			#pmid = '29686393'
			new_graph_obj.append([e1,e2,type_list,title_l[:self.topk_sent],pmid_l[:self.topk_sent],sent_l[:self.topk_sent]])

		return new_graph_obj
