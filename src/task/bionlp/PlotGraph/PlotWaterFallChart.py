import sys
import os
from shutil import copyfile
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
import operator

sys.path.append('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/bionlp/PlotGraph/')
sys.path.append('/oak/stanford/groups/rbaltman/swang91/Sheng_repo/src/task/bionlp/')
sys.path.append(repo_dir)
os.chdir(repo_dir)
import time
import collections
import numpy as np
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from plot_bionlp_figures import *
from utils import *
plt.switch_backend('agg')

plt.style.use('ggplot')

def read_term_net(dataset):
	if dataset == 'disease':
		file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/NLP_Dictionary/hp_obo_format.tsv'
	elif dataset == 'function':
		file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/NLP_Dictionary/GO_term.network'
	else:
		print 'wrong dataset',dataset
		sys.exit(-1)
	fin = open(file)
	net =  collections.defaultdict(dict)
	for line in fin:
		w = line.lower().strip().split('\t')
		net[w[0]][w[1]] = 1
	fin.close()
	return net

def return_term_child(net, term):
	child = set()
	new_child = set()
	new_child.add(term)
	child.add(term)
	while len(new_child)>0:
		newnew_child = set()
		for c in new_child:
			child.add(c)
			for ci in net[c]:
				if ci in child:
					continue
				newnew_child.add(ci)
		new_child = newnew_child
	return child

G2G_obj,network_gene_list = read_network_data(net_file_l = ['data/network/human/string_integrated.txt'])
ngene = len(network_gene_list)

dataset2name = {}
dataset2name['drug'] = 'Drug'
dataset2name['disease'] = 'Disease'
dataset2name['molecular_function'] = 'MF'
dataset2name['cellular_component'] = 'CC'
dataset2name['biological_process'] = 'BP'
for max_layer in [2,3,4,5]:
	for edge_wt_thres in [0.001,0.005,0.01,0.05,0.08,0.1]:

		ticks = []
		for dataset in ['function','disease']:#
			if dataset == 'disease':
				dataset_name = 'Monarch_Disease'
			elif dataset == 'drug':
				dataset_name = 'CTRP_GDSC_drugGene'
			elif dataset == 'function':
				dataset_name = 'Gene_Ontology'
			else:
				dataset_name = 'Gene_Ontology'

			candidate_gene,d2g,dataset_name = read_drug_disease_to_genes(dataset=dataset)
			ImproveNet_obj,ImproveNet_obj_baseline,stop_word_list = read_ImprovedNet_obj(dataset=dataset,min_freq_cutoff=4)
			term_net = read_term_net(dataset)

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
				#if tp!= dataset:
				#	continue
				our_rank = round(float(our_rank),4)
				base_rank = round(float(base_rank),4)
				dname = dname.replace('.txt','')
				our_method[dname] = our_rank + base_rank * 0.0001
				baseline_method[dname] = base_rank
				term_set.add(dname)
			fin.close()


			for term in term_set:
				child = return_term_child(term_net, term)
				our_bar = []
				base_bar = []
				tick = []
				our_method_subset = {}
				for c in child:
					if c not in our_method:
						continue
					our_method_subset[c] = our_method[c]
				our_method_l = sorted(our_method_subset.iteritems(), key=lambda (k,v): (v,k))
				our_method_l.reverse()
				for i in range(len(our_method_l)):
					our_bar.append(our_method_l[i][1])
					tick.append(our_method_l[i][0])
					base_bar.append(baseline_method[our_method_l[i][0]])
					if i>15:
						break

				b = np.array(base_bar)
				a = np.array(our_bar)
				if dataset=='function':
					if len(base_bar)<3 or np.mean(a)<0.5:
						continue
				else:
					if len(base_bar)<3 or np.mean(a)<0.5:
						continue
				if len(base_bar)>10:
					continue
				#print term,np.mean(a),np.mean(b),len(a)
				#print 'sb'
				gscore_mat = []
				tick = []
				auc_l = []
				nbar = 5
				nsample = 100
				width = 7 * nsample / 1000
				for c in child:
					if c not in our_method:
						continue
					if baseline_method[c] > our_method[c]:
						continue
					if our_method[c] < 0.8:
						continue
					pos_gene = set(d2g[c].keys())
					exist_pos_gene = set(network_gene_list) & set(pos_gene)
					if len(exist_pos_gene) == 0:
						continue
					if len(exist_pos_gene) > 100:
						continue

					FN_obj = FindNeighbor(c,ImproveNet_obj,stop_word_list=stop_word_list,exclude_edge_type = [],exclude_edges = [],include_genes=network_gene_list)
					g2score = FN_obj.CalNgh(G2G_obj,stop_word_list=stop_word_list,max_layer=max_layer,edge_wt_thres=edge_wt_thres,all_type_same_weight=False,use_direct_gene_edge=True)[0]
					g2score_sorted = sorted(g2score.iteritems(), key=lambda (k,v): (v,k))
					g2score_sorted.reverse()
					gscore_list = []
					for i in range(nsample):
						g = g2score_sorted[i][0]
						if g in exist_pos_gene:
							gscore_list.append(1)
						else:
							gscore_list.append(0)
					auc_l.append(our_method[c])
					if np.sum(gscore_list)==0:
						continue
					#print c
					gscore_mat.append(gscore_list)
					tick.append(c+' '+str(our_method[c]))
					if len(tick) > nbar:
						break
				if len(tick) < nbar:
					continue
				gscore_mat = np.array(gscore_mat)
				#print 'start'


				ind = np.arange(nsample)
				plt.clf()
				fig = plt.figure()
				# set height ratios for sublots
				gs = gridspec.GridSpec(nbar, 1)
				print nbar,term,auc_l
				# the fisrt subplot
				ax = {}
				ax[nbar-1] = plt.subplot(gs[nbar-1])
				ax[nbar-1].grid(False)
				# log scale for axis Y of the first subplot
				#ax0.set_yscale("log")
				#line0, = ax0.plot(x, y, color='r')
				ax[nbar-1].set_ylim(0, 1)

				ax[nbar-1].yaxis.set_ticks_position('none')
				plt.setp(ax[nbar-1].get_yticklabels(), visible=False)
				b1 = ax[nbar-1].bar(ind, gscore_mat[nbar-1,:nsample], width, color='#B47CC7')
				#the second subplot
				# shared axis X
				for ii in range(0,nbar-1):
					print ii
					ax[ii] = plt.subplot(gs[ii], sharex = ax[nbar-1])
					ax[ii].grid(False)
					ax[ii].set_ylim(0, 1)
					plt.setp(ax[ii].get_yticklabels(), visible=False)
					#line1, = ax1.plot(x, y, color='b', linestyle='--')
					b2 = ax[ii].bar(ind, gscore_mat[ii,:nsample], width, color='#B47CC7')
					yticks = ax[ii].yaxis.get_major_ticks()
					yticks[-1].label1.set_visible(False)
					plt.setp(ax[ii].get_xticklabels(), visible=False)
					ax[ii].yaxis.set_ticks_position('none')

				plt.subplots_adjust(hspace=.0)
				plt.tight_layout()
				plt.savefig(plot_result_dir+term+'_strip.pdf')

				ind = np.arange(len(our_bar))
				width = 0.7
				fig = plt.figure()
				ax = fig.add_subplot(111)
				ax = plt.subplot(1,1,1)
				b1 = ax.bar(ind, a, width, color='#B47CC7')
				b2 = ax.bar(ind+width*0.2, b,width, color='#6ACC65')#

				ax.set_title(term+' (n='+str(len(a))+')')
				ax.set_ylim(0.9,1.0)
				ax.set_xticks(ind+0.1)
				ax.set_ylabel('AUROC')
				#xticks(xticks_pos ,country_list, rotation=45 )
				ax.set_xticklabels(tick, rotation=90, fontsize =12)

				ax.legend((b1,b2), ('Our method', 'Baseline'))

				ax.xaxis.set_ticks_position("bottom")
				#ax2.yaxis.set_ticks_position("left")

				plt.subplots_adjust(hspace=.0)
				plt.tight_layout()
				plt.savefig(plot_result_dir+term+'_barplot.pdf')
