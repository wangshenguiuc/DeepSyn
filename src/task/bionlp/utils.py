import sys
import os
#repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'

#sys.path.append('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/bionlp/PlotGraph/')
#sys.path.append(repo_dir)
#os.chdir(repo_dir)
from src.models.network_flow.PlotNetworkFlow import plot_network_flow
import cPickle as pickle
from src.datasets.BioNetwork import BioNetwork
from src.datasets.FindNeighbor import FindNeighbor
from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.extractive_gen import ExtractGenSent
#from plot_bionlp_figures import *
import operator
import time
import collections
import numpy as np
import psutil
import pcst_fast
from src.utils.evaluate.evaluate import evaluate_vec

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


def read_ImprovedNet_obj(dataset='drug',min_freq_cutoff=5,max_ngh=20,read_baseline=True,DATA_DIR='',CACHE_DIR='',use_auto_phrase=False):
	#print dataset
	if dataset=='phrase' or dataset=='everything':
		use_auto_phrase = True
	#network_dump_file = CACHE_DIR + 'data/Pubmed/word_network/improved_word_net_190610_' + str(min_freq_cutoff)+'_'+dataset
	network_dump_file = CACHE_DIR + 'knowledge_graph_' + str(min_freq_cutoff)+'_'+dataset+'_'+str(use_auto_phrase)
	if max_ngh!=20:
		network_dump_file += '_'+str(max_ngh)
	if os.path.isfile(network_dump_file):
		ImproveNet_obj = pickle.load(open(network_dump_file, "rb" ))
	else:
		Net_obj = WordNet()
		pubmed_word_net = {DATA_DIR + 'data/Pubmed/word_network/predict_abst_181110':'infer',DATA_DIR + 'data/Pubmed/word_network/all_abst_181110':'pubmed'}
		Net_obj.ReadWordNet(pubmed_word_net,verbal=True,min_freq_cutoff=min_freq_cutoff)

		Net_obj.ReadWordType(use_auto_phrase=use_auto_phrase)
		stop_word_file = DATA_DIR + 'data/NLP_Dictionary/stopwords.txt'
		#stop_word_list = parse_word_net.GenerateStopWordList(Net_obj.word_ct,Net_obj.edge_ct,stop_word_file,Net_obj.word_type)
		stop_word_list = parse_word_net.GetStopWordList(stop_word_file)
		stop_word_list_manually = parse_word_net.GetStopWordList(DATA_DIR + 'data/NLP_Dictionary/stopwords_manually.txt')
		stop_word_list = stop_word_list.union(stop_word_list_manually)
		if dataset == 'disease':
			edge_list_l = [[dataset,'disease'],[dataset,'tisue'],[dataset,'gene'],[dataset,'function'],['tisue','function'],['tisue','disease'],['disease','function'],['disease','tisue'],['disease','disease'],['function','function'],
			['function','gene'],['tisue','gene'],['disease','gene'],['gene','gene'],['function','drug'],['gene','drug'],['disease','drug']]
			Net_obj.ReadEdgeType(stop_word_list, edge_list_l)
			#selected_kg_l = [Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2f]
			#selected_kg_l = [Net_obj.monarch_d2g,Net_obj.hpo_d2d,Net_obj.hpo_f2g,Net_obj.go_f2f,Net_obj.go_f2g]
			selected_kg_l = [Net_obj.Monarch_d2g,Net_obj.hpo_d2d,Net_obj.go_f2f,Net_obj.hpo_f2g,Net_obj.hpo_d2d,Net_obj.go_f2g,Net_obj.literome_g2g,Net_obj.PPI_g2g]
		elif dataset == 'drug':
			edge_list_l = [[dataset,'gene'],['gene',dataset]]
			Net_obj.ReadEdgeType(stop_word_list, edge_list_l)
			#selected_kg_l = [Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2f]
			#selected_kg_l = [Net_obj.monarch_d2g,Net_obj.hpo_d2d,Net_obj.hpo_f2g,Net_obj.go_f2f,Net_obj.go_f2g]
			selected_kg_l = [Net_obj.hpo_d2d,Net_obj.go_f2f,Net_obj.go_f2g]
		elif dataset == 'function':
			edge_list_l = [['function','gene'],['function','function']]
			Net_obj.ReadEdgeType(stop_word_list, edge_list_l)
			#selected_kg_l = [Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2f]
			#selected_kg_l = [Net_obj.monarch_d2g,Net_obj.hpo_d2d,Net_obj.hpo_f2g,Net_obj.go_f2f,Net_obj.go_f2g]
			selected_kg_l = [Net_obj.hpo_d2d,Net_obj.go_f2f]
		elif dataset == 'gene':
			edge_list_l = [['drug','gene'],['gene','drug']]
			Net_obj.ReadEdgeType(stop_word_list, edge_list_l)
			selected_kg_l = [Net_obj.go_f2g]
		elif dataset == 'phrase':
			edge_list_l = [['function','gene'],['function','function']]
			Net_obj.ReadEdgeType(stop_word_list, edge_list_l)
			#selected_kg_l = [Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2f]
			#selected_kg_l = [Net_obj.monarch_d2g,Net_obj.hpo_d2d,Net_obj.hpo_f2g,Net_obj.go_f2f,Net_obj.go_f2g]
			selected_kg_l = [Net_obj.hpo_d2d,Net_obj.go_f2f,Net_obj.go_f2g]
		elif dataset == 'everything':
			edge_list_l = [['disease','disease'],['disease','tisue'],['disease','gene'],['disease','function'],['tisue','function'],['tisue','disease'],['disease','function'],['disease','tisue'],['disease','disease'],['function','function'],
			['function','gene'],['tisue','gene'],['disease','gene']]
			Net_obj.ReadEdgeType(stop_word_list, edge_list_l)
			#selected_kg_l = [Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2f]
			#selected_kg_l = [Net_obj.monarch_d2g,Net_obj.hpo_d2d,Net_obj.hpo_f2g,Net_obj.go_f2f,Net_obj.go_f2g]
			selected_kg_l = [Net_obj.hpo_d2d,Net_obj.go_f2f,Net_obj.go_f2g]
		else:
			sys.exit('wrong dataset 1')
		ImproveNet_obj = ImprovedWordNet(Net_obj,selected_kg_l,max_ngh=max_ngh)
		Net_obj = WordNet()
		#print sys.getsizeof(ImprovedWordNet)
		#ImproveNet_obj.log_message('Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2g')
		#ImproveNet_obj.log_message('data/Pubmed/word_network/predict_abst_180717 data/Pubmed/word_network/all_abst_180305')

		ImproveNet_obj.reload()
		with open(network_dump_file, 'wb') as output:
		   pickle.dump(ImproveNet_obj, output, pickle.HIGHEST_PROTOCOL)


	stop_word_file = DATA_DIR + 'data/NLP_Dictionary/stopwords.txt'
	#stop_word_list = parse_word_net.GenerateStopWordList(Net_obj.word_ct,Net_obj.edge_ct,stop_word_file,Net_obj.word_type)
	stop_word_list = parse_word_net.GetStopWordList(stop_word_file)
	stop_word_list_manually = parse_word_net.GetStopWordList(DATA_DIR + 'data/NLP_Dictionary/stopwords_manually.txt')
	stop_word_list = stop_word_list.union(stop_word_list_manually)
	if read_baseline:
		baseline_network_dump_file = DATA_DIR + 'data/Pubmed/word_network/baseline_improved_word_net_181111_' + str(min_freq_cutoff)+'_'+dataset
		if os.path.isfile(baseline_network_dump_file):
			ImproveNet_obj_baseline = pickle.load(open(baseline_network_dump_file, "rb" ))
		else:
			edge_list_l = [[dataset,'gene']]
			Net_obj_baseline = WordNet()
			pubmed_word_net = {DATA_DIR + 'data/Pubmed/word_network/all_abst_181110':'pubmed'}
			Net_obj_baseline.ReadWordNet(pubmed_word_net,verbal=True,min_freq_cutoff=min_freq_cutoff)

			Net_obj_baseline.ReadWordType()
			Net_obj_baseline.ReadEdgeType(stop_word_list, edge_list_l)
			ImproveNet_obj_baseline = ImprovedWordNet(Net_obj_baseline,[])
			print sys.getsizeof(ImproveNet_obj_baseline)
			ImproveNet_obj_baseline.reload()
			with open(baseline_network_dump_file, 'wb') as output:
				pickle.dump(ImproveNet_obj_baseline, output, pickle.HIGHEST_PROTOCOL)
	else:
		ImproveNet_obj_baseline = ''

	return ImproveNet_obj,ImproveNet_obj_baseline,stop_word_list


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

def read_network_data(DATA_DIR='',net_file_l = ['data/network/human/string_integrated.txt'],network='string',rsp=0.8):

	G2G_obj = BioNetwork(net_file_l,weighted=True)
	literome_rwr_dump_file = DATA_DIR+'data/network/embedding/my_dca/'+network+'_RWR_'+str(rsp)
	G2G_obj.rwr = pickle.load(open(literome_rwr_dump_file, "rb" ))
	network_gene_list = []
	for g in G2G_obj.g2i.keys():
		network_gene_list.append(g.lower())
	return G2G_obj,network_gene_list

def read_drug_disease_to_genes(dataset='drug'):
	Net_obj_baseline = WordNet()
	if dataset == 'disease':
		dataset_name = 'Monarch_Disease'
		candidate_gene,old_d2g = Net_obj_baseline.ReadNodeTypeMonarchDiseaseGene()
		d2g = collections.defaultdict(dict)
		for ss in old_d2g:
			s = ss.replace('_',' ').replace('  ',' ')
			s = ''.join([i for i in s if not i.isdigit()]).strip()
			d2g[s] = old_d2g[ss]
		print len(d2g)
	elif dataset == 'drug':
		dataset_name = 'CTRP_GDSC_drugGene'
		#candidate_gene,d2g = Net_obj_baseline.ReadNodeTypeCtrpDrugGene()
		d2g = collections.defaultdict(dict)
		candidate_gene,d2g_gdsc = Net_obj_baseline.ReadNodeTypeCtrpDrugGene(file_name = 'data/drug/gdsc/drug_target_mapped.txt')
		candidate_gene,d2g_ctrp = Net_obj_baseline.ReadNodeTypeCtrpDrugGene(file_name = 'data/drug/ctrp/drug_target_formatted.txt')
		for d in d2g_gdsc:
			d2g[d] = d2g_gdsc[d]
		for d in d2g_ctrp:
			d2g[d] = d2g_ctrp[d]
		print len(d2g)
	elif dataset == 'function':
		dataset_name = 'Gene_Ontology'
		#candidate_gene,d2g = Net_obj_baseline.ReadNodeTypeCtrpDrugGene()
		candidate_gene = set()
		d2g = Net_obj_baseline.ReadEdgeTypeGO2Gene()
		for d in d2g:
			for g in d2g[d]:
				candidate_gene.add(g)
		print len(d2g)
	else:
		sys.exit('wrong dataset')
	return candidate_gene,d2g,dataset_name

def create_clean_dir(result_dir):
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	else:
		for subdir, dirs, files in os.walk(result_dir):
			for file in files:
				file_path = os.path.join(subdir, file)
				os.remove(file_path)
