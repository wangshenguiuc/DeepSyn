import multiprocessing
import sys
import os
import collections
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
sys.path.append(repo_dir)
os.chdir(repo_dir)

from src.datasets.WordNet import WordNet,parse_sentence
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.preprocess_word_sent import get_word_path, write_word_sents_to_file


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


print 'number of words',len(word_set)

fout = open(word_list_file,'w')
for w in word_set:
	fout.write(w.lower()+'\n')
fout.close()
