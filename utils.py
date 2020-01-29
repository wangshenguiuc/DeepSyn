import numpy as np
import sys
import os
#os.chdir(repo_dir)
from collections import Counter
import operator
import time
import collections
import copy
import numpy as np
from scipy import stats
from scipy.sparse import isspmatrix
from scipy.sparse import csr_matrix

layer_keys = ['drug', 'gene','function', 'disease']


def extract_network(net_obj, ki, kj):
	net = net_obj.bp_net
	net_wt = {}
	#for src in net:
	#	max_wt  = -1
	for n1 in net:
		for n2 in net[n1]:
			if n1 not in net_obj.word_type or n2 not in net_obj.word_type:
				continue
			if (wtype(net_obj, n1) == ki and wtype(net_obj, n2) == kj) or (wtype(net_obj, n1) == kj and wtype(net_obj, n2) == ki):
				for src in net[n1][n2]:
					wt = net[n1][n2][src]
					if src not in net_wt:
						net_wt[src] = collections.defaultdict(dict)
					net_wt[src][n1][n2] = wt
		#for n1 in net_wt[src]:
		#	for n2 in net_wt[src][n1]:
		#		net_wt[src][n1][n2] /= wt
	return net_wt

def wtype(net_obj, n):
	tp = net_obj.word_type[n].split('_')[1]
	return tp


def impute_query(key, keu, network):
	ki = layer_keys.index(key)
	if key!='disease':
		upp_ngh = find_ngh(network, querys[querys[ki+1]], key)
	if key!='drug':
		upp_layer = querys[ki-1]
		low_ngh = find_ngh(network, querys[querys[ki-1]], key)
	ngh = upp_ngh
	ngh.extend(low_ngh)
	return ngh

def find_ngh(network, starts, tp, node2tp, topk=3):
	terms = []
	for s in starts:
		if node2tp[s] == tp:
			terms.append(s)
			continue
		wts = {}
		for ngh in network[s]:
			src, wt = network[s][ngh]
			wts[ngh] = wts.get(ngh, 0) + wt
		wts_sort = sorted(wts.items(), key=lambda x: x[1], reverse=True)
		for i in range(min(topk, len(wts_sort))):
			terms.append(wts_sort[i][0])
	terms = np.unique(terms)
	return terms

def search_path(networks, starts, ends, max_depth=2, verbal=False):

	ends = set(ends)
	starts = set(starts)
	paths = set()
	find_ends = set()
	for s in starts:
		pre = {}
		pre[s] = ''
		curs = set()
		curs.add(s)
		cur_ends = ends.copy()
		visited = set()
		for d in range(max_depth):
			for s in curs:
				cur_ends.discard(s)
				visited.add(s)
			if len(cur_ends)==0:
				break
			new_curs = set()
			for s in curs:
				for ngh in networks[s]:
					if ngh in visited or ngh in new_curs:
						continue
					new_curs.add(ngh)
					pre[ngh] = s
			curs = new_curs
		for e in ends:
			if e in cur_ends:
				if verbal:
					print ('no path found',e)
				continue
			find_ends.add(e)
			x = e
			while pre[x]!='':
				paths.add(pre[x]+'\t'+x+'\t'+networks[pre[x]][x][0])
				x = pre[x]
	return paths, find_ends


def read_server_data(DATA_DIR, MAX_DEPTH = 4, NNODES_PER_LAYER = 5, NNODES_PER_QUERY = 3):

	deep_syn_network = DATA_DIR
	diffusion = collections.defaultdict(dict)
	diffusion_n2i = collections.defaultdict(dict)
	diffusion_i2n = collections.defaultdict(dict)

	nkeys = len(layer_keys)
	for depth in range(1,5):
		if depth!=MAX_DEPTH:
			continue
		for ki in ['gene','function']:
			kj = ki
			if False and os.path.isfile(deep_syn_network_large +  ki + '_' + kj + '_'+str(depth-1)+ '_diffusion.npy'):
				diffusion_file = deep_syn_network_large +  ki + '_' + kj+ '_'+str(depth-1) + '_diffusion.npy'
				diffusion_n2i_file = deep_syn_network_large +  ki + '_' + kj+ '_'+str(depth-1) + '_diffusion_n2i.npy'
				diffusion_i2n_file = deep_syn_network_large +  ki + '_' + kj + '_'+str(depth-1)+'_diffusion_i2n.npy'
			else:
				diffusion_file = deep_syn_network +  ki + '_' + kj + '_'+str(depth-1) + '_diffusion.npy'
				diffusion_n2i_file = deep_syn_network +  ki + '_' + kj + '_'+str(depth-1) + '_diffusion_n2i.npy'
				diffusion_i2n_file = deep_syn_network +  ki + '_' + kj + '_'+str(depth-1) + '_diffusion_i2n.npy'
			diffusion[depth][ki] = np.load(diffusion_file, allow_pickle=True)
			diffusion_n2i[depth][ki] = np.load(diffusion_n2i_file, allow_pickle=True).item()
			diffusion_i2n[depth][ki] = np.load(diffusion_i2n_file, allow_pickle=True).item()

	node2tp = {}
	tp2node = {}
	node2ct = {}
	for k in layer_keys:
		tp2node[k] = set()
	networks = collections.defaultdict(dict)

	nkeys = len(layer_keys)
	for i in range(nkeys-1):
		ki = layer_keys[i]
		for j in range(i,max(i+2,nkeys)):
			kj = layer_keys[j]
			networks[ki][kj] = collections.defaultdict(dict)
			for src in ['pubmed','infer','database']:
				file = deep_syn_network + src + '_'+ki+'_'+kj+'_unnormalized.txt'

				if not os.path.isfile(file):
					continue
				fin = open(file)
				for line in fin:
					w = line.strip().split('\t')
					n1,tp1,ct1,n2,tp2,ct2,wt = w
					ct1 = int(ct1)
					ct2 = int(ct2)
					tp1 = tp1.split('_')[1]
					tp2 = tp2.split('_')[1]

					tp2node[tp1].add(n1)
					tp2node[tp2].add(n2)
					node2tp[n1] = tp1
					node2tp[n2] = tp2
					node2ct[n1] = max(node2ct.get(n1,0),ct1)
					node2ct[n2] = max(node2ct.get(n2,0),ct2)
					if src=='database':
						node2ct[n1] += 1
						node2ct[n2] += 1
					networks[ki][kj][n1][n2] = [src, float(wt)]
					networks[ki][kj][n2][n1] = [src, float(wt)]
				fin.close()

	return diffusion, diffusion_n2i, diffusion_i2n, networks, node2tp, tp2node, node2ct

def term2termid(terms, n2i):
	term_ids = []
	for term in terms:
		if term in n2i:
			term_ids.append(n2i[term])
	term_ids = np.unique(np.array(term_ids))
	return term_ids

def forward_search(dk, ki, kj, org_query, query, network_st_ed, network_st_mid, network_mid_ed, diffusion, diffusion_n2i, diffusion_i2n, node2tp, MAX_DEPTH, NNODES_PER_LAYER, NNODES_PER_QUERY):
	print (dk, ki, kj)
	starts = query[ki]
	ends = query[kj]
	terms = starts.copy()
	terms.extend(ends)
	print ('qkj',query[kj])
	print ('starts',starts)
	if len(starts) == 0 and len(query[kj]) != 0:
		starts = find_ngh(network_st_ed, query[kj].copy(), ki, node2tp, topk=NNODES_PER_QUERY)
		middles = query[kj].copy()
	else:
		middles = find_ngh(network_st_ed, starts, dk, node2tp, topk=NNODES_PER_QUERY)
	print ('starts',starts)
	print ('qki',query[ki])
	print ('middles',middles)
	middle_ids = term2termid(middles, diffusion_n2i[MAX_DEPTH][dk])

	if len(middle_ids)==0 or len(org_query[ki])==0: # no diffusion
		diff_terms = middles
	else:
		vec = np.sum(diffusion[MAX_DEPTH][dk][middle_ids,:], axis = 0)
		diff_terms = [diffusion_i2n[MAX_DEPTH][dk][j] for j in np.argsort(vec*-1)[:NNODES_PER_LAYER]]

	diff_terms = np.unique(diff_terms)

	print ('starts',starts)
	print ('diff_terms',diff_terms,len(org_query[ki]),org_query[ki])
	print ('ends',ends)
	if set(diff_terms) == set(starts):
		start_paths = set()
	else:
		print (ki,dk)
		start_paths, diff_terms = search_path(network_st_mid, starts, diff_terms, max_depth=MAX_DEPTH, verbal=False)
	print ('start_paths',start_paths)
	print ('diff_terms',diff_terms)

	depi = 0
	if len(ends)>0:
		end_paths = set()
		find_ends = []
		while (len(end_paths)==0 or len(find_ends)!=len(ends)):
			end_paths, find_ends = search_path(network_mid_ed, diff_terms, ends, max_depth=MAX_DEPTH+depi, verbal=False)
			depi += 1
			if depi>4:
				break
	else:
		end_paths = set()
		find_ends = diff_terms
	path = start_paths | end_paths
	print ('end_paths',end_paths)
	print ('find_ends',ends)
	return path, find_ends


def run_query(query, networks, diffusion, diffusion_n2i, diffusion_i2n, node2tp, MAX_DEPTH = 5, NNODES_PER_LAYER = 5, NNODES_PER_QUERY = 3):
	org_query = copy.deepcopy(query)
	layer_keys = ['drug', 'gene','function', 'disease']
	diffusion_keys = ['gene','function','function']
	paths = set()
	nkeys = len(layer_keys)
	for i in range(0, nkeys-1):
		dk = diffusion_keys[i]
		ki = layer_keys[i]
		kj = layer_keys[i+1]

		if ki=='drug' or ki=='gene':
			layer_paths, next_layer_terms = forward_search(dk, ki, kj, org_query, query, networks[ki][kj], networks[ki][dk], networks[dk][kj], diffusion, diffusion_n2i, diffusion_i2n, node2tp, MAX_DEPTH, NNODES_PER_LAYER, NNODES_PER_QUERY)
		else:
			#ki function, kj disease
			rev_layer_paths, next_layer_terms = forward_search(dk, kj, ki, org_query, query, networks[ki][kj], networks[dk][kj], networks[ki][dk], diffusion, diffusion_n2i, diffusion_i2n, node2tp, MAX_DEPTH, NNODES_PER_LAYER, NNODES_PER_QUERY)
			layer_paths = set()
			for p in rev_layer_paths:
				w1,w2,tp=p.split('\t')
				layer_paths.add(w2+'\t'+w1+'\t'+tp)
		paths = paths | layer_paths
		print (next_layer_terms)
		query[kj].extend(list(next_layer_terms))
		query[kj] = list(np.unique(query[kj]))
		print (query)

	paths = list(paths)
	nodes = set()
	for path in paths:
		w1, w2, tp = path.split('\t')
		nodes.add(w1)
		nodes.add(w2)
	return paths, nodes
