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
from src.utils.evaluate.evaluate import evaluate_vec

plt.switch_backend('agg')


go2cat = {}
fin = open('data/function_annotation/GO2name.txt')
for line in fin:
	w  =line.lower().strip().split('\t')
	if len(w)<2:
		continue
	go2cat[w[1]] = w[2]
fin.close()

for dataset in ['disease','function']:
	if dataset == 'disease':
		dataset_name = 'Monarch_Disease'
	elif dataset == 'drug':
		dataset_name = 'CTRP_GDSC_drugGene'
	elif dataset == 'function':
		dataset_name = 'Gene_Ontology'
	else:
		sys.exit('wrong dataset')


	#AUROC	macrothrombocytopenia	0.907802975086	0.625	0.149333639582	0.250794281176	0.3	0.4	16

	plot_result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/bionlp/figures/'
	if not os.path.exists(plot_result_dir):
		os.makedirs(plot_result_dir)

	for max_layer in [2,3,4,5]:
		for edge_wt_thres in [0.001,0.005,0.01,0.05,0.08,0.1]:
			print max_layer,edge_wt_thres
			result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow_tmp/'+dataset_name+'/'+str(max_layer)+'_'+str(edge_wt_thres)+'/'
			base_f2f = collections.defaultdict(dict)
			our_f2f = collections.defaultdict(dict)
			for i,dir in enumerate(os.listdir(result_dir)):
				cur_file = result_dir + '/' + dir
				#print cur_file
				fin = open(cur_file)
				for line in fin:
					f1,f2,r1,r2,br = line.strip().split('\t')
					base_f2f[f1][f2] = float(br)
					our_f2f[f1][f2] = float(r2)
					#print f1,f2
				fin.close()
				#if i%500==0 and i>100:
				#	print i,len(os.listdir(result_dir))
				#	break
			base_auc_l = []
			our_auc_l = []
			fout = open(plot_result_dir+dataset_name+'_'+str(max_layer)+'_'+str(edge_wt_thres),'w')
			for f in base_f2f:
				label = []
				our_score = []
				base_score = []
				for g in base_f2f[f]:
					if f==g:
						label.append(1)
					else:
						label.append(0)
					our_score.append(our_f2f[f][g])
					base_score.append(base_f2f[f][g])
				if np.sum(label)==0:
					continue
				base_auc,pear,spear,auprc,prec_at_k = evaluate_vec(base_score,label)
				our_auc,pear,spear,auprc,prec_at_k = evaluate_vec(our_score,label)
				base_auc_l.append(base_auc)
				our_auc_l.append(our_auc)
				if dataset == 'function':
					if f in go2cat:
						fout.write(f+'\t'+go2cat[f]+'\t'+str(base_auc)+'\t'+str(our_auc)+'\n')
				else:
					fout.write(f+'\t'+dataset+'\t'+str(base_auc)+'\t'+str(our_auc)+'\n')
				#print np.mean(base_auc_l), np.mean(our_auc_l),base_auc,our_auc
			fout.close()
