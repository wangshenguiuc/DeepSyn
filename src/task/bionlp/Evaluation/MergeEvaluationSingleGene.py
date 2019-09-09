import sys
import os
from shutil import copyfile
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'


sys.path.append(repo_dir)
os.chdir(repo_dir)
import time
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')


go2cat = {}
fin = open('data/function_annotation/GO2name.txt')
for line in fin:
	w  =line.lower().strip().split('\t')
	if len(w)<2:
		continue
	go2cat[w[1]] = w[2]
fin.close()

for dataset in ['function']:
	if dataset == 'disease':
		dataset_name = 'Monarch_Disease'
	elif dataset == 'drug':
		dataset_name = 'CTRP_GDSC_drugGene'
	elif dataset == 'function':
		dataset_name = 'Gene_Ontology'
	else:
		sys.exit('wrong dataset')

	plot_result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/figures/'
	if not os.path.exists(plot_result_dir):
		os.makedirs(plot_result_dir)
	#AUROC	macrothrombocytopenia	0.907802975086	0.625	0.149333639582	0.250794281176	0.3	0.4	16

	for max_layer in [2,3,4,5]:
		for edge_wt_thres in [0.001,0.005,0.01,0.05,0.08,0.1]:
			result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'+str(max_layer)+'_'+str(edge_wt_thres)+'/'
			our_auc_l = []
			base_auc_l = []
			our_auprc_l = []
			base_auprc_l = []
			our_prec_l = []
			base_prec_l = []
			en_l = []
			for i,dir in enumerate(os.listdir(result_dir)):
				cur_file = result_dir + '/' + dir
				if not os.path.isfile(cur_file):
					#print cur_file
					continue
				#print cur_file
				fin = open(cur_file)
				w = fin.readline().strip().split('\t')
				if len(w)<7:
					#print w
					continue
				if float(w[2])==0.5 and float(w[3])==0.5:
					continue
				our_auc_l.append(float(w[2]))
				base_auc_l.append(float(w[3]))
				our_auprc_l.append(float(w[4]))
				base_auprc_l.append(float(w[5]))
				our_prec_l.append(float(w[6]))
				base_prec_l.append(float(w[7]))
				en_l.append(dir.replace('.txt',''))
				fin.close()
				#if i%2000==0:
				#	print i
			print max_layer,edge_wt_thres,len(base_prec_l),np.mean(our_auc_l),np.mean(base_auc_l),np.mean(our_auprc_l),np.mean(base_auprc_l),np.mean(our_prec_l),np.mean(base_prec_l)
			nfunc = len(our_auc_l)
			fout = open(plot_result_dir+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres)+'_single_gene','w')
			for i in range(len(our_auc_l)):
				if dataset == 'function':
					if en_l[i] not in go2cat:
						print en_l[i]
						continue
					fout.write(en_l[i]+'\t'+go2cat[en_l[i]]+'\t'+str(base_auc_l[i])+'\t'+str(our_auc_l[i])+'\n')
				else:
					fout.write(en_l[i]+'\t'+dataset+'\t'+str(base_auc_l[i])+'\t'+str(our_auc_l[i])+'\n')
			fout.close()
			plt.clf()
			plt.scatter(base_auc_l,our_auc_l)
			plt.xlabel('AUROC of baseline')
			plt.ylabel('AUROC of our method')
			plt.xlim([0,1])
			plt.ylim([0,1])
			plt.plot( [0,1],[0,1] , ls="--", c=".3")
			plt.title(dataset)
			#plt.show()
			plt.savefig(plot_result_dir+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres)+'_single_gene.pdf')
