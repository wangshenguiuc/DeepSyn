from nltk.stem.porter import *
import nltk
import os
import subprocess
import operator
import collections
from nltk.stem.wordnet import WordNetLemmatizer
from src.models.generate_sentence.preprocess_word_sent import get_word_pair_path

class preprocess_parser_tree():

	def __init__(self,nlp_tool_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/tools/stanford_parser_java/stanford-parser-full-2017-06-09/',topk_key_word=2):
		self.nlp_tool_dir = nlp_tool_dir
		self.topk_key_word = topk_key_word
		self.lmtzr = WordNetLemmatizer()
		self.stemmer = PorterStemmer()

	def get_path_from_node(self,st,net):
		cur = st
		path = [st.split('-')[0]]
		while cur!='ROOT-0':
			next = net[cur].keys()[0]
			path.append(next.split('-')[0])
			cur = next
		return path

	def get_path(self,edge,st,ed):
		nedge = len(edge)
		net = collections.defaultdict(dict)
		for e in edge:
			tp = e.split('(')[0]
			e1 = e.split('(')[1].split(',')[0].lstrip(' ')
			e2 = e.split('(')[1].split(',')[1].split(')')[0].lstrip(' ')
			if st == e2.split('-')[0]:
				st = e2
			if ed == e2.split('-')[0]:
				ed = e2
			#e1 = e1.split('-')[0].lstrip(' ')
			#e2 = e2.split('-')[0].lstrip(' ')
			net[e2][e1] = tp
		if st not in net or ed not in net:
			return 'Stanford NLP tool error',[]

		st_path = self.get_path_from_node(st,net)
		ed_path = self.get_path_from_node(ed,net)

		find = False
		for si in range(0,len(st_path)):
			for ei in range(0,len(ed_path)):
				if st_path[si]==ed_path[ei]:
					find = True
					break
			if find:
				break
		path = ''
		word_list = []
		for i in range(0,si):
			path += st_path[i] + ' '
			if st_path[i] != st.split('-')[0]:
				word_list.append(self.lmtzr.lemmatize(self.stemmer.stem(st_path[i].decode("utf8"))))
		for i in range(ei,-1,-1):
			path += ed_path[i] + ' '
			if ed_path[i]!=ed.split('-')[0]:
				word_list.append(self.lmtzr.lemmatize(self.stemmer.stem(ed_path[i].decode("utf8"))))
		return path, word_list

	def parse_stanford_NLP_output(self,new_w1,new_w2,output_path):
		fin = open(output_path)
		edge = []
		path_l = []
		vocab_dct = {}
		stem_word_list_l = []
		for line in fin:
			text = line.strip()
			if text == '':
				path,stem_word_list = self.get_path(edge,new_w2,new_w1)
				stem_word_list_l.append(stem_word_list)
				path_l.append(path)
				for w in stem_word_list:
					vocab_dct[w] = vocab_dct.get(w,0) + 1
				edge = []
			else:
				edge.append(text)
		fin.close()

		vocab_list = sorted(vocab_dct.items(),key=operator.itemgetter(1))
		vocab_list.reverse()
		vocab_score = {}
		for ct,i in enumerate(vocab_list):
			if ct > self.topk_key_word:
				break
			vocab_score[i[0]] = i[1]
		return vocab_score,path_l,stem_word_list_l

	def get_parser_tree(self,w1,w2,sent_l, pid=0, use_cache=True):
		word_path_dir,word = get_word_pair_path(w1,w2)
		word_path = word_path_dir + word
		new_w2 = w2.replace(' ','_')
		new_w1 = w1.replace(' ','_')
		word_path_input = word_path + '_parser.input'
		word_path_output = word_path + '_parser.output'
		if os.path.isfile(word_path_output) and use_cache:
			vocab_score,path_l,stem_word_list_l = self.parse_stanford_NLP_output(new_w1,new_w2,word_path_output)
			if len(path_l) > 0:
				return vocab_score,path_l,stem_word_list_l

		fout = open(word_path_input,'w')
		for s in sent_l:
			new_s = s.lower().translate(None,',?!:()=%>/[]').strip().replace('.','')+' .'
			fout.write(new_s.rstrip().lstrip().replace(w2,new_w2).replace(w1,new_w1)+'\n')
		fout.close()

		cur_dir = os.getcwd()
		os.chdir(self.nlp_tool_dir)
		command = 'java -Xmx2g -cp "*" edu.stanford.nlp.parser.nndep.DependencyParser -model edu/stanford/nlp/models/parser/nndep/english_UD.gz -textFile '+'"'+word_path_input+'"'+' -outFile '+'"'+word_path_output+'"'
		output = subprocess.check_output(command, shell=True)
		vocab_score,path_l,stem_word_list_l = self.parse_stanford_NLP_output(new_w1,new_w2,word_path_output)
		os.chdir(cur_dir)
		return vocab_score,path_l,stem_word_list_l
