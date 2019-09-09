import sys
import os
from configure import hyperpara
repo_dir = hyperpara['repo_dir']
sys.path.append(repo_dir + 'src/task/bionlp/')
sys.path.append(repo_dir)
os.chdir(repo_dir)
import cPickle as pickle
from src.datasets.BioNetwork import BioNetwork
from src.datasets.FindLayer import FindLayer
from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.extractive_gen import ExtractGenSent
from utils import *
import operator
import time
import collections
import numpy as np
import psutil
import networkx as nx
import pcst_fast
from src.utils.evaluate.evaluate import evaluate_vec
from src.models.network_flow.PlotNetworkFlow import plot_network_flow

DATA_DIR = '/data/cellardata/users/netant/Data/'
CACHE_DIR = '/data/cellardata/users/netant/Cache/'
SenGene_obj = ExtractGenSent(DATA_DIR = DATA_DIR,CACHE_DIR = CACHE_DIR)

e1 = 'glucokinase activity'
e2 = 'diabetes mellitus'
graph_sent_list = SenGene_obj.get_graph_sentence([[e1,e2,'','infer']],'')
for e1,e2,type_list,title_l,pmid_l,sent_l in graph_sent_list:
	print e1,e2,title_l[0],pmid_l[0],sent_l[0]
	
e1 = 'breast cancer'
e2 = 'lung cancer'
graph_sent_list = SenGene_obj.get_graph_sentence([[e1,e2,'','infer']],'')
for e1,e2,type_list,title_l,pmid_l,sent_l in graph_sent_list:
	print e1,e2,title_l[0],pmid_l[0],sent_l[0]
