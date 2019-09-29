import numpy as np
import operator
from scipy import stats

def GenerateStopWordList(word_ct,edge_ct,stop_word_file,node_type,reference_node_type = 'hpo_disease',min_freq_cutoff=10,ntop_candidate=50000,ignore_node_type=[]):
	stop_words = set()
	gene_list = []
	#ignore_node_type = ['cancer_gene','cancer_disease','literome_gene', 'barbasi_tissue','CCLE_gene', 'mesh_disease', 'mesh_tissue','hpo_disease','GO_function','ncbi_gene']

	for g in node_type:
		if node_type[g] == reference_node_type and g in word_ct and word_ct[g]>min_freq_cutoff:
			gene_list.append(g)
	fout = open(stop_word_file,'w')
	sort_x = sorted(word_ct.items(),key=operator.itemgetter(1))
	sort_x.reverse()
	new_d = {}
	for i in range(ntop_candidate):
		w = sort_x[i][0]
		p_list = []
		q_list = []
		for g in gene_list:
			if g in edge_ct[w]:
				p_list.append(edge_ct[w][g]*1.0)
			else:
				p_list.append(0.)
			q_list.append(word_ct[g]*1.0)
		p_list = np.array(p_list) / np.sum(np.array(p_list))
		q_list = np.array(q_list) / np.sum(np.array(q_list))
		kl = stats.entropy(p_list,q_list)
		new_d[w] = kl
	sort_x = sorted(new_d.items(),key=operator.itemgetter(1))
	for i in range(ntop_candidate):
		w = sort_x[i][0]
		#print kl,np.sum(q_list),np.sum(p_list)
		if w not in node_type:
			node_type[w] = 'unknown'
		if node_type[w] in ignore_node_type:
			continue
		fout.write(w+'\t'+node_type[w]+'\t'+str(sort_x[i][1])+'\t'+str(word_ct[w])+'\n')
		stop_words.add(w)
	fout.close()
	return GetStopWordList(stop_word_file)

def GetStopWordList(stop_word_file,MI_cutoff=0.9):
	stop_words = set()
	fin = open(stop_word_file)
	for line in fin:
		w = line.lower().strip().split('\t')[0]
		if len(line.strip().split('\t')) == 1:
			stop_words.add(w)
			continue
		sc = float(line.strip().split('\t')[2])
		if sc <= MI_cutoff:
			continue
		stop_words.add(w)
	fin.close()
	return stop_words
