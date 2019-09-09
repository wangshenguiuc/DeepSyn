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

#plt.style.use('ggplot')

def set_box_color(bp, color):
	plt.setp(bp['boxes'], color=color)
	plt.setp(bp['whiskers'], color=color)
	plt.setp(bp['caps'], color=color)
	plt.setp(bp['medians'], color=color)
plt.style.use('dark_background')
plot_result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/large_box_plot/'
if not os.path.exists(plot_result_dir):
	os.makedirs(plot_result_dir)
dataset2name = {}
dataset2name['drug'] = 'Drug'
dataset2name['disease'] = 'Dis'
dataset2name['molecular_function'] = 'MF'
dataset2name['cellular_component'] = 'CC'
dataset2name['biological_process'] = 'BP'
for max_layer in [2,3,4,5]:
	print max_layer
	for edge_wt_thres in [0.001,0.005,0.01,0.05,0.08,0.1]:
		print edge_wt_thres
		our_method = {}
		baseline_method = {}
		ticks = []
		for dataset in ['function','disease','drug']:
			if dataset == 'disease':
				dataset_name = 'Monarch_Disease'
			elif dataset == 'drug':
				dataset_name = 'CTRP_GDSC_drugGene'
			elif dataset == 'function':
				dataset_name = 'Gene_Ontology'
			else:
				sys.exit('wrong dataset')

			file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/figures/'+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres)+'_single_gene'
			#prefix = ' (gene)'
			prefix =''
			if not os.path.isfile(file):
				print file
				continue
			fin = open(file)
			for line in fin:
				dname,tp,base_rank, our_rank = line.strip().split('\t')
				dcat = dataset2name[tp] + prefix
				#base_rank_l.append(float(base_rank))
				#our_rank_l.append(float(our_rank))
				if dcat not in ticks:
					ticks.append(dcat)
				if dcat not in our_method:
					our_method[dcat] = []
				if dcat not in baseline_method:
					baseline_method[dcat] = []
				our_method[dcat].append(float(our_rank))
				baseline_method[dcat].append(float(base_rank))
			fin.close()
			#print dataset,len(our_method),len(baseline_method)

			file = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/figures/'+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres)
			prefix = ' (set)'
			if not os.path.isfile(file):
				print file
				continue
			fin = open(file)
			#print file
			for line in fin:
				#print line
				dname,tp,base_rank, our_rank = line.strip().split('\t')
				dcat = dataset2name[tp]  + prefix
				#base_rank_l.append(float(base_rank))
				#our_rank_l.append(float(our_rank))
				if dcat not in ticks:
					ticks.append(dcat)
				if dcat not in our_method:
					our_method[dcat] = []
				if dcat not in baseline_method:
					baseline_method[dcat] = []
				our_method[dcat].append(float(our_rank))
				baseline_method[dcat].append(float(base_rank))
			fin.close()

		ticks = ['BP','MF','CC','BP (set)','MF (set)','CC (set)','Drug','Dis','Dis (set)']
		our_method_l = []
		base_method_l = []
		ticks = list(ticks)
		ticks_l = []
		for t in ticks:
			our_method_l.append(our_method[t])
			base_method_l.append(baseline_method[t])
			ticks_l.append(t+"\n(n="+str(len(our_method[t]))+')')
			#ticks_l.append(t)
			#print t,len(our_method[t]),len(baseline_method[t])

		bg_color = 'black'
		fg_color = 'white'

		fig = plt.figure()
		width = 0.6
		bpl = plt.boxplot(our_method_l, positions=np.array(xrange(len(our_method)))*2.0-0.4+1, sym='', widths=0.6,patch_artist=True)
		bpr = plt.boxplot(base_method_l, positions=np.array(xrange(len(baseline_method)))*2.0+0.4+1, sym='', widths=0.6,patch_artist=True)
		set_box_color(bpl, '#F9E00D') # colors are from http://colorbrewer2.org/
		set_box_color(bpr, '#3F6FB6')
		#for box in bpl['boxes']:
		#	box.set_facecolor('#D7191C' )
		#for box in bpr['boxes']:
		#	box.set_facecolor( '#2C7BB6' )

		# draw temporary red and blue lines and use them to create a legend
		plt.plot([], c='#F9E00D', label='Our method')
		plt.plot([], c='#3F6FB6', label='Baseline')
		plt.legend(loc='best')

		plt.xticks(xrange(1, len(ticks_l) * 2+1, 2), ticks_l)
		#plt.xlim(-2, len(ticks)*2)
		plt.ylim(0.1, 1.1)
		plt.xlim(0, len(ticks_l) * 2+1)
		plt.tight_layout()
		plt.savefig(plot_result_dir+'_'+str(max_layer)+'_'+str(edge_wt_thres)+'.pdf')
		#break
