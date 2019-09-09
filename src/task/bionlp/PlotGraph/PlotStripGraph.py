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
from matplotlib import gridspec
import matplotlib.pyplot as plt
from plot_bionlp_figures import *
from utils import *
plt.switch_backend('agg')
plt.style.use('dark_background')
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

plot_result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/waterfall_plot/SingleGene/'
if not os.path.exists(plot_result_dir):
	os.makedirs(plot_result_dir)

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
for max_layer in [2]:
	#for edge_wt_thres in [0.001,0.005,0.01,0.05,0.08,0.1]:
	for edge_wt_thres in [0.01]:
		fin = open('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/function_score/phrase/'+str(max_layer)+'_'+str(edge_wt_thres)+'/1.txt')
		background_sc = []
		for line in fin:
			sc = line.strip().split('\t')[2]
			background_sc.append(float(sc))
		background_sc = np.array(background_sc)


		plot_result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/stripe_plot/'+dataset_name+'/'+str(max_layer)+'_'+str(edge_wt_thres)+'/'
		if not os.path.exists(plot_result_dir):
			os.makedirs(plot_result_dir)

		file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/figures/'+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres)+'_single_gene'
		#prefix = ' (gene)'
		prefix =''
		if not os.path.isfile(file):
			print file
			continue
		fin = open(file)
		our_method = {}
		baseline_method = {}
		term_set = set()
		for line in fin:
			dname,tp,base_rank, our_rank = line.strip().split('\t')
			our_rank = round(float(our_rank),4)
			base_rank = round(float(base_rank),4)
			dname = dname.replace('.txt','')
			our_method[dname] = our_rank + base_rank * 0.0001
			baseline_method[dname] = base_rank
		fin.close()

		#print our_method

		for ci,ss in enumerate(d2g.keys()):
			#continue

			print ci,len(d2g.keys())
			if '/' in ss:
				ss = ss.replace('/','')
			if ss!='fanconi anemia':
				continue
			if ss not in our_method:
				continue
			if ci%total_pid != pid and total_pid>1 :
				continue
			if ss not in ImproveNet_obj.net:
				continue
			pos_gene = set(d2g[ss].keys())
			exist_pos_gene = set(network_gene_list) & set(pos_gene)
			if len(exist_pos_gene) == 0:
				continue
			if our_method[ss]<0.9 or baseline_method[ss]>our_method[ss]:
				continue

			result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'+str(max_layer)+'_'+str(edge_wt_thres)+'/'
			#create_clean_dir(result_dir)
			if not os.path.exists(result_dir):
				os.makedirs(result_dir)

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
			if np.sum(label)==0:
				continue
			gsum = np.sum(g2score.values())
			nsample = 50
			auc,pear,spear,auprc,prec_at_k = evaluate_vec(score,label)
			base_auc,pear,spear,base_auprc,base_prec_at_k = evaluate_vec(baseline_score,label)

			g2score_sorted = sorted(g2score.iteritems(), key=lambda (k,v): (v,k))
			g2score_sorted.reverse()
			a  = []
			posi = []
			tick = []
			posg = []
			for i in range(nsample):
				a.append(g2score_sorted[i][1])
				if g2score_sorted[i][0] in exist_pos_gene:
					posi.append(i)
					posg.append(g2score_sorted[i][0])
				tick.append(g2score_sorted[i][0].upper())

			gscore_list = []
			for i in range(nsample):
				g = g2score_sorted[i][0]
				if g in exist_pos_gene:
					gscore_list.append(1)
				else:
					gscore_list.append(0)

			if dataset=='function':
				if len(posi)<=5 or posi[0]>2:
					continue
			else:
				if len(posi)<=3 or posi[0]>2:
					continue

			a = np.array(a)
			ind = np.arange(len(a)) + 1
			width = 0.55
			plt.clf()
			fig = plt.figure(figsize=(8, 6))
			# set height ratios for sublots
			gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
			ax = {}


			width = 1 * nsample / 1000
			ax[1] = plt.subplot(gs[0])
			barlist = ax[1].bar(ind, gscore_list, width, color='red')
			ax[1].set_xticks(ind)
			ax[1].grid(False)
			ax[1].set_title(ss+'('+str(auc)+')'+str(np.sum(gscore_list)))
			ax[1].set_ylim(0, 1)

			#ax[1].yaxis.set_ticks_position('none')
			plt.setp(ax[1].get_yticklabels(), visible=False)
			plt.setp(ax[1].get_xticklabels(), visible=False)
			#ax[1].set_xticklabels(tick, rotation=90, fontsize =4)

			pv = []
			for i in range(nsample):
				print len(np.where(background_sc>a[i])[0]) + 1,a[i]
				pv.append((len(np.where(background_sc>a[i])[0]) + 1+i) * 1.0 / (len(background_sc)+nsample))
			pv = np.array(pv)
			pv = np.log10(pv)*-1
			#print pv


			ax[0] = plt.subplot(gs[2], sharex = ax[1])
			width = 0.5
			barlist = ax[0].bar(ind-width/2., pv, width, color='#FAF118')
			ax[0].set_ylim(0,np.floor(np.max(pv))+1)
			#ax[0].set_xticks(ind)
			#ax[0].set_xlim(0, np.max(ind+2))
			ax[0].set_yticks(np.arange(0, np.floor(np.max(pv))+1, 1.0))
			ax[0].set_ylabel('-log(P-value)')
			ax[0].set_xticklabels(tick, rotation=90, fontsize =8)
			ax[0].xaxis.set_ticks_position("bottom")
			ax[0].set_xlim(0, np.max(ind)+1)

			print ss,auc,gscore_list,posg,pv
			ax[2] = plt.subplot(gs[1])
			ac = np.expand_dims(pv, axis=0)*-1
			ax[2].imshow(ac, cmap=plt.cm.bwr, aspect='auto')
			plt.setp(ax[2].get_yticklabels(), visible=False)
			plt.setp(ax[2].get_xticklabels(), visible=False)
			#plt.setp(ax[0].get_xticklabels(), visible=False)

			plt.tight_layout()
			plt.savefig(plot_result_dir+ss+'.pdf')
			#plt.savefig('tmp.pdf')
			#sys.exit(-1)
