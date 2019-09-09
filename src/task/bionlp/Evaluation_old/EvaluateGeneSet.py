import sys
import os
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'

sys.path.append('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/bionlp/PlotGraph/')
sys.path.append('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/bionlp/')
sys.path.append(repo_dir)
os.chdir(repo_dir)
from src.models.network_flow.PlotNetworkFlow import plot_network_flow
import cPickle as pickle
from src.datasets.BioNetwork import BioNetwork
from src.datasets.FindNeighbor import FindNeighbor
from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.extractive_gen import ExtractGenSent
from src.models.pathway_enrichment.PathwayDrugResponse import PathwayDrugResponse
from plot_bionlp_figures import *
from utils import *
import operator
import time
import collections
import numpy as np
import psutil
import pcst_fast
from src.utils.evaluate.evaluate import evaluate_vec


if len(sys.argv) <= 2:
	pid = 1
	total_pid = 1
	#dataset = 'disease'
else:
	pid = int(sys.argv[1])
	#dataset = str(sys.argv[2])
	total_pid = int(sys.argv[2])

dataset = 'disease'
d2g_exp,d2g_mut = read_drug_mute_exp_data()

ImproveNet_obj,ImproveNet_obj_baseline,stop_word_list = read_ImprovedNet_obj(dataset=dataset,min_freq_cutoff=4)

Net_obj = WordNet()
go_f2f = Net_obj.ReadEdgeTypeGO()
GO_ngh_set = set()
for f in go_f2f:
	for ngh1 in go_f2f[f]:
		for ngh2 in go_f2f[f]:
			GO_ngh_set.add(ngh1+'#'+ngh2)

candidate_gene,d2g,dataset_name = read_drug_disease_to_genes(dataset=dataset)

print dataset_name

SenGene_obj = ExtractGenSent(working_dir = repo_dir)

G2G_obj,network_gene_list = read_network_data(net_file_l = ['data/network/human/string_integrated.txt'])


for topk in [2]:
	for max_layer in [4,5]:
		for edge_wt_thres in [0.01,0.03,0.05,0.08]:
			result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'+str(max_layer)+'_'+str(edge_wt_thres)+'_'+str(topk)+'/'
			if not os.path.exists(result_dir):
				os.makedirs(result_dir)
			for ci,ss in enumerate(d2g.keys()):
				#if ci<0.95*len(d2g.keys()):
				#	continue
				if ci%total_pid != pid and total_pid>1 :
					continue
				s = ss.replace('_',' ').replace('  ',' ')

				if dataset_name == 'Monarch_Disease':
					s = ''.join([i for i in s if not i.isdigit()]).strip()
				if s not in ImproveNet_obj.net:
					continue
				#print s

				pos_gene = set(d2g[s].keys())
				exist_pos_gene = set(network_gene_list) & set(pos_gene)
				if len(exist_pos_gene) == 0:
					#print pos_gene
					continue



				#for g in exist_pos_gene:
				#	fout.write(g+'\n')

				FN_obj = FindNeighbor(s,ImproveNet_obj,stop_word_list=stop_word_list,exclude_edge_type = [],exclude_edges = GO_ngh_set)
				g2score = FN_obj.CalNgh(G2G_obj,stop_word_list=stop_word_list,max_layer=max_layer,edge_wt_thres=edge_wt_thres)
				#quality_Score, quality_Score_detail= FN_obj.CalQuality(FN_obj.node_set,FN_obj.edge_list)
				#network_output_file = result_dir + s
				#plot_network_flow(network_output_file,s,FN_obj.node_set,FN_obj.edge_list,FN_obj.l2nodes, ImproveNet_obj.word_ct, ImproveNet_obj.word_type)

				output_file = result_dir + s+'_gset'
				node_set, edge_list, node_weight = FN_obj.GetSubNetwork(set(exist_pos_gene),topk=topk)
				#print node_set, edge_list, node_weight
				g_quality_Score, g_quality_Score_detail= FN_obj.CalQuality(node_set,edge_list)
				nselect_node = FN_obj.nselect_node
				g_nfunc,g_nlayer,g_ngene,g_ngene_ngh = g_quality_Score_detail

				if g_nfunc<=4 or g_nfunc>=25 or g_ngene<=3 or g_ngene>=10:
					continue
				if nselect_node*1./g_ngene>0.8:
					continue
				file = result_dir+s+'.txt'
				fout = open(file,'w')
				fout.write(str(g_nfunc)+'\t'+str(g_nlayer)+'\t'+str(g_ngene)+'\t'+str(g_ngene_ngh)+'\t'+str(nselect_node)+'\n')
				print g_nfunc,g_nlayer,g_ngene,g_ngene_ngh,nselect_node

				plot_network_flow(output_file,s,node_set,edge_list,node_weight, ImproveNet_obj.word_ct, ImproveNet_obj.word_type,tgt_set = exist_pos_gene)
				os.remove(output_file)
				print ci,ci*1.0/len(d2g.keys()),len(exist_pos_gene),max_layer,edge_wt_thres,s
				fout.close()
