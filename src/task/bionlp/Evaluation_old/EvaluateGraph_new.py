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
	dataset = 'disease'
else:
	pid = int(sys.argv[1])
	dataset = str(sys.argv[2])
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


for topk in [10]:
	for max_layer in [4,5]:
		for edge_wt_thres in [0.08]:
			for ci,ss in enumerate(d2g.keys()):
				#if ci<0.95*len(d2g.keys()):
				#	continue
				if ci%total_pid != pid and total_pid>1 :
					continue
				s = ss.replace('_',' ').replace('  ',' ')
				if dataset_name == 'Monarch_Disease':
					s = ''.join([i for i in s if not i.isdigit()]).strip()
				if s not in ImproveNet_obj.net or s not in d2g_exp:
					continue
				print s

				pos_gene = set(d2g[s].keys())
				exist_pos_gene = set(network_gene_list) & set(pos_gene)
				if len(exist_pos_gene) == 0:
					continue
				result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'+str(max_layer)+'_'+str(edge_wt_thres)+'/' +s + '/'+str(topk)+'/'
				create_clean_dir(result_dir)

				file = result_dir+s+'.txt'
				fout = open(file,'w')

				FN_obj = FindNeighbor(s,ImproveNet_obj,stop_word_list=stop_word_list,exclude_edge_type = [],exclude_edges = GO_ngh_set)
				g2score = FN_obj.CalNgh(G2G_obj,stop_word_list=stop_word_list,max_layer=max_layer,edge_wt_thres=edge_wt_thres)
				quality_Score, quality_Score_detail= FN_obj.CalQuality(FN_obj.node_set,FN_obj.edge_list)
				network_output_file = result_dir + s
				plot_network_flow(network_output_file,s,FN_obj.node_set,FN_obj.edge_list,FN_obj.l2nodes, ImproveNet_obj.word_ct, ImproveNet_obj.word_type)

				sorted_x = sorted(g2score.items(), key=operator.itemgetter(1))
				sorted_x.reverse()
				import_genes = []
				waterfall_sc = []
				for i in range(topk):
					g = sorted_x[i][0]
					if i>topk and g not in pos_gene:
						#print i,sorted_x[i][0],sorted_x[i][1], g in pos_gene
						#fout.write(str(i)+' '+str(sorted_x[i][0])+' '+str(sorted_x[i][1])+' '+str(int(g in pos_gene))+'\n')
						continue
					i#f g not in pos_gene:
					import_genes.append(g.upper())
					mut_score = 1
					exp_score = 0
					if (s in d2g_exp and g in d2g_exp[s]) or (s in d2g_mut and g in d2g_mut[s]):
						if s in d2g_mut:
							mut_score = d2g_mut[s].get(g,1)
						if s in d2g_exp:
							exp_score = d2g_exp[s].get(g,0)

					output_file = result_dir + s+'_'+g
					node_set, edge_list, node_weight = FN_obj.GetSubNetwork(set([g]))
					g_quality_Score, g_quality_Score_detail= FN_obj.CalQuality(node_set,edge_list)
					if g_quality_Score_detail[1]<=2:
						continue

					plot_network_flow(output_file,s,node_set,edge_list,node_weight, ImproveNet_obj.word_ct, ImproveNet_obj.word_type)
					bs = -1
					if s in ImproveNet_obj_baseline.net and g in ImproveNet_obj_baseline.net[s]:
						bs = ImproveNet_obj_baseline.net[s][g]['pubmed']
					#print i,sorted_x[i][0], bs, g in pos_gene, mut_score,exp_score,g_quality_Score,g_quality_Score_detail
					fout.write(str(i)+'\t'+str(sorted_x[i][0])+'\t'+str(sorted_x[i][1])+'\t'+str(int(g in pos_gene))+'\t'+str( exp_score)+'\t'+str( mut_score)+'\t'+str(g_quality_Score)+'\t'+str(g_quality_Score_detail[1])+'\n')

				# calculate AUROC for baseline and our method
				score = []
				baseline_score = []
				label = []
				for g in g2score:
					bs = -1
					if s in ImproveNet_obj_baseline.net and g in ImproveNet_obj_baseline.net[s]:
						bs = ImproveNet_obj_baseline.net[s][g]['pubmed']
					score.append(g2score[g])
					baseline_score.append(bs)
					label.append(int(g in pos_gene))
				auc,pear,spear,auprc = evaluate_vec(score,label)
				base_auc,pear,spear,base_auprc = evaluate_vec(baseline_score,label)
				print ci,ci*1.0/len(d2g.keys()),max_layer,edge_wt_thres,s,auc,base_auc,quality_Score,quality_Score_detail
				fout.write('AUROC\t'+str(s)+'\t'+str(auc)+'\t'+str(base_auc)+'\t'+str(auprc)+'\t'+str(base_auprc)+'\t'+str(np.sum(label))+'\t'+str(quality_Score)+'\n')

				fout.flush()
				fout.close()
