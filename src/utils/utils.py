import sys
import os
import pickle
from utils.BioNetwork import BioNetwork
#from plot_bionlp_figures import *
import operator
import time
import collections
import numpy as np
import psutil
from utils.WordNet import WordNet
from utils.ImprovedWordNet import ImprovedWordNet
from utils import parse_word_net
def write_graph_edges(edge_list,sent_file):

	for e1,e2,sc,source in edge_list:
		fout = open(sent_file+e1+'_'+e2+'.txt','w')
		sr = source.split('_')
		if 'pubmed' in sr:
			sent_l = get_sentence_edge(e1,e2)
			fout.write(e1+'\t'+e2+'\n')
			for sent in sent_l:
				fout.write(sent+'\n')
			fout.write('\n')
		elif 'infer' in sr:
			sent_l = get_inferred_sentence_edge(e1,e2)
			fout.write(e1+'\t'+e2+'\n')
			for sent in sent_l:
				fout.write(sent+'\n')
			fout.write('\n')
		fout.close()


def read_ImprovedNet_obj(pubmed_file, deepsyn_file, DATA_DIR, min_freq_cutoff=10, dataset='disease', max_ngh = 10 ,use_cache=True):
	network_dump_file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/software/DeepSyn_data/knowledge_graph'
	if os.path.isfile(network_dump_file) and use_cache:
		ImproveNet_obj = pickle.load(open(network_dump_file, "rb" ))
		stop_word_file = DATA_DIR + 'NLP_Dictionary/stopwords.txt'
		stop_word_list = parse_word_net.GetStopWordList(stop_word_file)
		stop_word_list_manually = parse_word_net.GetStopWordList(DATA_DIR + 'NLP_Dictionary/stopwords_manually.txt')
		stop_word_list = stop_word_list.union(stop_word_list_manually)
	else:
		pubmed_word_net = {deepsyn_file:'infer',pubmed_file:'pubmed'}
		Net_obj = WordNet(DATA_DIR = DATA_DIR)
		Net_obj.ReadWordNet(pubmed_word_net,verbal=False,min_freq_cutoff=min_freq_cutoff)
		Net_obj.ReadWordType(use_auto_phrase=True)


		edge_list_l = [[dataset,'disease'],[dataset,'tisue'],[dataset,'gene'],[dataset,'function'],['tisue','function'],['tisue','disease'],['disease','function'],['disease','tisue'],['disease','disease'],['function','function'],
		['function','gene'],['tisue','gene'],['disease','gene'],['gene','gene'],['function','drug'],['gene','drug'],['disease','drug']]
		Net_obj.ReadEdgeType(stop_word_list, edge_list_l)
		selected_kg_l = [Net_obj.Monarch_d2g,Net_obj.hpo_d2d,Net_obj.go_f2f,Net_obj.hpo_f2g,Net_obj.hpo_d2d,Net_obj.go_f2g,Net_obj.literome_g2g,Net_obj.PPI_g2g]

		ImproveNet_obj = ImprovedWordNet(Net_obj,selected_kg_l,max_ngh=max_ngh)
		ImproveNet_obj.reload()
		with open(network_dump_file, 'wb') as output:
		   pickle.dump(ImproveNet_obj, output, pickle.HIGHEST_PROTOCOL)
	return ImproveNet_obj, stop_word_list


def read_drug_mute_exp_data(data_method = ['ccle','gdsc']):

	d2g_exp = {}
	d2g_mut = {}
	drug_cor_cutoff=0.3
	for method in data_method:
		fin = open('data/cell_line_data/'+method+'top_genes_exp_hgnc.txt')
		for line in fin:
			w = line.lower().strip().split('\t')
			cor = float(w[2])
			if abs(cor) < drug_cor_cutoff:
				continue
			d = w[1]
			if d not in d2g_exp:
				d2g_exp[d] = {}
			d2g_exp[d][w[0]] = float(w[2])
		fin.close()

		fin = open(repo_dir+'data/cell_line_data/'+method+'_sign_drug_mut.txt')
		for line in fin:
			w = line.lower().strip().split('\t')
			d = w[1]
			if d not in d2g_mut:
				d2g_mut[d] = {}
			d2g_mut[d][w[0]] = float(w[3])
		fin.close()
	return d2g_exp,d2g_mut

def read_network_data(net_file_l,network_rwr):
	G2G_obj = BioNetwork([net_file_l],weighted=True)
	G2G_obj.rwr = pickle.load(open(network_rwr, "rb" ), encoding='latin1')
	network_gene_list = []
	for g in G2G_obj.g2i.keys():
		network_gene_list.append(g.lower())
	return G2G_obj,network_gene_list


def create_clean_dir(result_dir):
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	else:
		for subdir, dirs, files in os.walk(result_dir):
			for file in files:
				file_path = os.path.join(subdir, file)
				os.remove(file_path)
