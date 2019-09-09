import sys
import os
from shutil import copyfile
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
import operator

sys.path.append(repo_dir)
os.chdir(repo_dir)
import time
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

plt.style.use('ggplot')


for dataset in ['function','disease','drug']:
	if dataset == 'disease':
		dataset_name = 'Monarch_Disease'
	elif dataset == 'drug':
		dataset_name = 'CTRP_GDSC_drugGene'
	elif dataset == 'function':
		dataset_name = 'Gene_Ontology'
	else:
		sys.exit('wrong dataset')


	#AUROC	macrothrombocytopenia	0.907802975086	0.625	0.149333639582	0.250794281176	0.3	0.4	16

	plot_result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/figures/'+dataset+'_to_gene/'
	if not os.path.exists(plot_result_dir):
		os.makedirs(plot_result_dir)

	for max_layer in [2,3,4,5]:
		for edge_wt_thres in [0.001,0.005,0.01,0.05,0.08,0.1]:
			file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/figures/'+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres)+'_single_gene'
			if not os.path.isfile(file):
				print file
				continue
			fin = open(file)

			base_rank_l = []
			our_rank_l = []
			for line in fin:
				base_rank, our_rank = line.strip().split('\t')
				base_rank_l.append(float(base_rank))
				our_rank_l.append(float(our_rank))
			fin.close()
			print len(base_rank_l),len(our_rank_l)
			data_to_plot = [np.array(base_rank_l), np.array(our_rank_l)]
			plt.clf()
			fig = plt.figure(1, figsize=(9, 6))
			# Create an axes instance
			ax = fig.add_subplot(111)
			ax.set_title(dataset)
			bp = ax.boxplot(data_to_plot)
			# Save the figure
			ax.set_xticklabels(['baseline','our method'])
			ax.set_ylabel('AUROC')
			fig.savefig(plot_result_dir+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres)+'boxplot.pdf', bbox_inches='tight')
			plt.clf()
			fig = plt.figure(1, figsize=(9, 6))
			# Create an axes instance
			ax = fig.add_subplot(111)
			ax.set_title(dataset)
			ax.set_ylabel('AUROC')
			# Create the boxplot
			bp = ax.violinplot(data_to_plot)
			#ax.set_xticklabels(['baseline','our method'])
			# Save the figure
			plt.setp(ax, xticks=[y+1 for y in range(2)],xticklabels=['baseline','our method'])
			fig.savefig(plot_result_dir+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres)+'violinplot.pdf', bbox_inches='tight')


	plot_result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/figures/'+dataset+'_to_gene_set/'
	if not os.path.exists(plot_result_dir):
		os.makedirs(plot_result_dir)

	for max_layer in [2,3,4,5]:
		for edge_wt_thres in [0.001,0.005,0.01,0.05,0.08,0.1]:
			file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/figures/'+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres)
			if not os.path.isfile(file):
				print file
				continue
			fin = open(file)

			base_rank_l = []
			our_rank_l = []
			for line in fin:
				base_rank, our_rank = line.strip().split('\t')
				base_rank_l.append(float(base_rank))
				our_rank_l.append(float(our_rank))
			fin.close()
			print len(base_rank_l),len(our_rank_l)
			data_to_plot = [np.array(base_rank_l), np.array(our_rank_l)]
			plt.clf()
			fig = plt.figure(1, figsize=(9, 6))
			ax = fig.add_subplot(111)
			ax.set_title(dataset)
			ax.set_xticklabels(['baseline','our method'])
			ax.set_ylabel('Rank of gene sets')
			bp = ax.boxplot(data_to_plot)
			fig.savefig(plot_result_dir+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres)+'boxplot.pdf', bbox_inches='tight')

			plt.clf()
			fig = plt.figure(1, figsize=(9, 6))
			ax = fig.add_subplot(111)
			ax.set_title(dataset)
			#ax.set_xticklabels(['baseline','our method'])
			ax.set_ylabel('Rank of gene sets')
			bp = ax.violinplot(data_to_plot)
			plt.setp(ax, xticks=[y+1 for y in range(2)],xticklabels=['baseline','our method'])
			fig.savefig(plot_result_dir+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres)+'violinplot.pdf', bbox_inches='tight')

