import sys
import os
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'

sys.path.append(repo_dir)
os.chdir(repo_dir)
from src.models.network_flow.PlotNetworkFlow import plot_network_flow
import cPickle as pickle
from src.datasets.FindNeighbor import FindNeighbor
from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.extractive_gen import ExtractGenSent
from src.models.generate_sentence.preprocess_word_sent import write_word_sents_to_file,get_sentence_edge,get_sentence_word,get_word_path,sentence_edge_is_in_cache
from src.models.generate_sentence.preprocess_parser_tree import preprocess_parser_tree
import operator
import time
import collections
import numpy as np
import psutil
from src.utils.evaluate.evaluate import evaluate_vec

e1 = 'dna demethylation'
for e2 in ['dna demethylase activity']:
	print e2,'-------------'
	sent_l = get_sentence_edge(e1,e2)
	for sent in sent_l:
		print sent
