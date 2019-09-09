import sys
import os
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'


sys.path.append(repo_dir)
os.chdir(repo_dir)

import cPickle as pickle
from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.extractive_gen import ExtractGenSent
from src.datasets.SubGraph import SubGraph
from src.datasets.KHopGraph import KHopGraph
import operator
import time
import collections
import numpy as np
import psutil
from src.utils.evaluate.evaluate import evaluate_vec


use_gene_2_gene = int(sys.argv[1])
dataset = str(sys.argv[2])
use_direct_edge = int(sys.argv[3])

min_freq_cutoff = 12


stop_word_file = 'data/NLP_Dictionary/stopwords.txt'
#stop_word_list = parse_word_net.GenerateStopWordList(Net_obj.word_ct,Net_obj.edge_ct,stop_word_file,Net_obj.word_type)
stop_word_list = parse_word_net.GetStopWordList(stop_word_file)
stop_word_list_manually = parse_word_net.GetStopWordList('data/NLP_Dictionary/stopwords_manually.txt')
stop_word_list = stop_word_list.union(stop_word_list_manually)


network_dump_file = 'data/Pubmed/word_network/improved_word_net_180814_' + str(min_freq_cutoff)+'_'+dataset+'_'+str(use_direct_edge)+'_'+str(use_gene_2_gene)
if os.path.isfile(network_dump_file):
    print 'find',network_dump_file
    ImproveNet_obj = pickle.load(open(network_dump_file, "rb" ))
    print 'find',network_dump_file
else:
    #
    Net_obj = WordNet()
    pubmed_word_net = {'data/Pubmed/word_network/predict_abst_180814':'infer','data/Pubmed/word_network/all_abst_180305':'pubmed'}
    Net_obj.ReadWordNet(pubmed_word_net,verbal=True,min_freq_cutoff=min_freq_cutoff)

    Net_obj.ReadWordType()
    edge_list_l = [[dataset,'symptom'],[dataset,'tissue'],[dataset,'disease'],[dataset,'function'],['tissue','function'],['tissue','symptom'],['symptom','function'],['symptom','tissue'],
        ['function','function'],['function','gene']]
    if use_direct_edge:
        edge_list_l.append([dataset,'gene'])
    if use_gene_2_gene:
        edge_list_l.append(['gene','gene'])
    Net_obj.ReadEdgeType(stop_word_list, edge_list_l)

    selected_kg_l = [Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2f,Net_obj.go_f2g]

    ImproveNet_obj = ImprovedWordNet(Net_obj,selected_kg_l)
    print sys.getsizeof(ImprovedWordNet)
    ImproveNet_obj.log_message('Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2')
    ImproveNet_obj.log_message('data/Pubmed/word_network/predict_abst_180717 data/Pubmed/word_network/all_abst_180305')
    ImproveNet_obj.reload()
    with open(network_dump_file, 'wb') as output:
        pickle.dump(ImproveNet_obj, output, pickle.HIGHEST_PROTOCOL)
