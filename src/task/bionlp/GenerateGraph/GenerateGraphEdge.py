
import sys
import os
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
predict_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/Pubmed/GO_sentences_L2_all/'
pubmed_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/Pubmed/pubmed/'
sys.path.append(repo_dir)
os.chdir(repo_dir)

from src.datasets.WordNet import WordNet
from src.datasets.WordNet import parse_sentence
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
import operator
import collections

pubmed_word_net = 'data/Pubmed/word_network/all_abst_181127'
predict_word_net = 'data/Pubmed/word_network/predict_abst_181127'

word_list_file = 'data/NLP_Dictionary/all_words.txt'

stop_word_file = 'data/NLP_Dictionary/stopwords.txt'
stop_word_list = parse_word_net.GetStopWordList(stop_word_file)
stop_word_list_manually = parse_word_net.GetStopWordList('data/NLP_Dictionary/stopwords_manually.txt')
stop_word_list = stop_word_list.union(stop_word_list_manually)

word_set = set()
fin = open(word_list_file)
for line in fin:
	word_set.add(line.strip().lower())
fin.close()


stop_word_file = 'data/NLP_Dictionary/stopwords.txt'
#stop_word_list = parse_word_net.GenerateStopWordList(Net_obj.word_ct,Net_obj.edge_ct,stop_word_file,Net_obj.word_type)
stop_word_list = parse_word_net.GetStopWordList(stop_word_file)

stop_word_list_manually = parse_word_net.GetStopWordList('data/NLP_Dictionary/stopwords_manually.txt')
stop_word_list = stop_word_list.union(stop_word_list_manually)

word_list = set(word_set).difference(stop_word_list)

pubmed_net = {}
wct = {}
sent_ct = 0
fout = open(predict_word_net,'w')
ct = 0
for file in os.listdir(predict_dir):
	GO = file.split('.txt')[0]
	if GO not in pubmed_net:
		pubmed_net[GO] = {}
	fin = open(predict_dir + file)
	for line in fin:
		sent_ct += 1
		text = line.lower().translate(None,',?!:()=%>/[]').strip().split('\t')[0]
		pset = parse_sentence(word_list,text,max_phrase_length = 8, move_sub_string=False)
		for p in pset:
			wct[p] = wct.get(p,0) + 1
			wct[GO]  = wct.get(GO,0) + 1
			pubmed_net[GO][p] = pubmed_net[GO].get(p,0) + 1
	fin.close()
	ct += 1
	print ct
for w1 in pubmed_net:
	for w2 in pubmed_net[w1]:
		fout.write(w1.replace('_',' ')+'\t'+str(wct[w1])+'\t'+w2+'\t'+str(wct[w2])+'\t'+str(pubmed_net[w1][w2])+'\t'+str(sent_ct)+'\n')
fout.close()

pubmed_net = {}
wct = {}
fout = open(pubmed_word_net,'w')
sent_ct = 0
ct = 0
for file in os.listdir(pubmed_dir):
	fin = open(pubmed_dir + file)
	for line_l in fin:
		for line in line_l.split('. '):
			sent_ct+=1
			text = line.lower().translate(None,'.,?!:()=%>/[]').strip().split('\t')[0]
			pset = parse_sentence(word_list,text,max_phrase_length = 8, move_sub_string=False)
			for p1 in pset:
				for p2 in pset:
					if p1>=p2:
						continue
					if p1 not in pubmed_net:
						pubmed_net[p1] = {}
					pubmed_net[p1][p2] = pubmed_net[p1].get(p2,0) + 1
					wct[p1] = wct.get(p1,0)+1
					wct[p2] = wct.get(p2,0)+1
	fin.close()
	ct += 1
	print ct
for w1 in pubmed_net:
	for w2 in pubmed_net[w1]:
		fout.write(w1+'\t'+str(wct[w1])+'\t'+w2+'\t'+str(wct[w2])+'\t'+str(pubmed_net[w1][w2])+'\t'+str(sent_ct)+'\n')
fout.close()
