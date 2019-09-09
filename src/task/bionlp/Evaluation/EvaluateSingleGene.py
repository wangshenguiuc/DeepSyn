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


'''
disease -> function -> gene set -> drug
1. disease, rank gene.
2. drug, rank gene.
3. gene set -> GO
4. GWAS -> GO
5. GO, rank gene
'''
if len(sys.argv) <= 2:
	pid = 1
	total_pid = 1
	dataset = 'disease'
else:
	dataset = str(sys.argv[1])
	pid = int(sys.argv[2])
	total_pid = int(sys.argv[3])


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

SenGene_obj = ExtractGenSent(working_dir = repo_dir)

G2G_obj,network_gene_list = read_network_data(net_file_l = ['data/network/human/string_integrated.txt'])


print len(network_gene_list)
for max_layer in [2,3,4,5]:
	for edge_wt_thres in [0.001,0.005,0.01,0.05,0.08,0.1]:
		for ci,ss in enumerate(d2g.keys()):
			#continue
			if '/' in ss:
				ss = ss.replace('/','')
			if ci%total_pid != pid and total_pid>1 :
				continue
			if ss not in ImproveNet_obj.net:
				continue
			pos_gene = set(d2g[ss].keys())
			exist_pos_gene = set(network_gene_list) & set(pos_gene)
			if len(exist_pos_gene) == 0:
				continue



			result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'+str(max_layer)+'_'+str(edge_wt_thres)+'/'
			#create_clean_dir(result_dir)
			if not os.path.exists(result_dir):
				os.makedirs(result_dir)
			file = result_dir+ss+'.txt'
			fout = open(file,'w')

			FN_obj = FindNeighbor(ss,ImproveNet_obj,stop_word_list=stop_word_list,exclude_edge_type = [],exclude_edges = [],include_genes=network_gene_list)
			g2score = FN_obj.CalNgh(G2G_obj,stop_word_list=stop_word_list,max_layer=max_layer,edge_wt_thres=edge_wt_thres,all_type_same_weight=False,use_direct_gene_edge=True)[0]

			score = []
			baseline_score = []
			label = []
			for g in g2score:
				bs = -1
				if ss in ImproveNet_obj_baseline.net and g in ImproveNet_obj_baseline.net[ss]:
					bs = ImproveNet_obj_baseline.net[ss][g]['pubmed']
				score.append(g2score[g])
				baseline_score.append(bs)
				label.append(int(g in exist_pos_gene))
			gsum = np.sum(g2score.values())
			#if gsum==0:

				#print ss
				#print ImproveNet_obj.net[ss]
				#for ngh in ImproveNet_obj.net[ss]:
				#	print ss,ngh,ImproveNet_obj.word_type[ngh],ImproveNet_obj.net[ss][ngh],ngh in network_gene_list
				#	if ngh not in ImproveNet_obj.net:
				#		continue
				#	for ngh1 in ImproveNet_obj.net[ngh]:
				#		print ngh1,ImproveNet_obj.word_type[ngh1]

				#sys.exit(-1)
			auc,pear,spear,auprc,prec_at_k = evaluate_vec(score,label)
			base_auc,pear,spear,base_auprc,base_prec_at_k = evaluate_vec(baseline_score,label)
			print ci,ci*1.0/len(d2g.keys()),max_layer,edge_wt_thres,ss,auc,base_auc,prec_at_k,base_prec_at_k,np.sum(label)
			fout.write('AUROC\t'+str(ss)+'\t'+str(auc)+'\t'+str(base_auc)+'\t'+str(auprc)+'\t'+str(base_auprc)+'\t'+str(prec_at_k)+'\t'+str(base_prec_at_k)+'\t'+str(np.sum(label))+'\n')
			fout.flush()
			fout.close()
