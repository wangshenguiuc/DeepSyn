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
import pickle
import networkx as nx

layer_keys = ['drug', 'gene','function', 'disease']

def remove_non_db_terms(query, tp2node):
	new_query = {}
	for key in query:
		new_query[key] = []
		value = query[key]
		for v in value:
			if v in tp2node[key]:
				new_query[key].append(v)
	return new_query

def valid_query(query, tp2node):
	for key in query:
		value = query[key]
		for v in value:
			if v in tp2node[key]:
				return True
	return False

def create_networkx_obj(paths, nodes, node2tp):
	G = nx.DiGraph()
	for node in nodes:
		G.add_node(node, type=node2tp[node])
	for path in paths:
		e1,e2,tp = path.split('\t')
		G.add_edge(e1, e2, type=tp)
	return G

def write_to_cyto_scape(paths, nodes, node2tp, output_file):
	fout = open(output_file+'.edge','w')
	fout.write('source\ttarget\ttype\n')
	for path in paths:
		e1,e2,tp = path.split('\t')
		fout.write(e1+'\t'+e2+'\t'+tp+'\n')
	fout.close()

	fout = open(output_file+'.node','w')
	fout.write('node\ttype\n')
	for n in nodes:
		fout.write(n+'\t'+node2tp[n]+'\n')
	fout.close()

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
	starts = query[ki]
	ends = query[kj]
	terms = starts.copy()
	terms.extend(ends)
	if len(starts) == 0 and len(query[kj]) != 0:
		starts = find_ngh(network_st_ed, query[kj].copy(), ki, node2tp, topk=NNODES_PER_QUERY)
		middles = query[kj].copy()
	else:
		middles = find_ngh(network_st_ed, starts, dk, node2tp, topk=NNODES_PER_QUERY)

	middle_ids = term2termid(middles, diffusion_n2i[MAX_DEPTH][dk])

	if len(middle_ids)==0 or len(org_query[ki])==0: # no diffusion
		diff_terms = middles
	else:
		vec = np.sum(diffusion[MAX_DEPTH][dk][middle_ids,:], axis = 0)
		diff_terms = [diffusion_i2n[MAX_DEPTH][dk][j] for j in np.argsort(vec*-1)[:NNODES_PER_LAYER]]

	diff_terms = np.unique(diff_terms)

	if set(diff_terms) == set(starts):
		start_paths = set()
	else:
		start_paths, diff_terms = search_path(network_st_mid, starts, diff_terms, max_depth=MAX_DEPTH, verbal=False)
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
		query[kj].extend(list(next_layer_terms))
		query[kj] = list(np.unique(query[kj]))


	paths = list(paths)
	nodes = set()
	for path in paths:
		w1, w2, tp = path.split('\t')
		nodes.add(w1)
		nodes.add(w2)
	return paths, nodes

def read_node_info(DATA_DIR):

	fin = open(DATA_DIR+'/node_info.txt')
	info = {}
	for line in fin:
		w = line.strip().split('\t')
		if len(w)<3:
			continue
		n,i,u=w
		info[n] = [[n,i,u]]
	fin.close()

	with open(DATA_DIR+'/pid2term.pickle', 'rb') as handle:
	    term2pid = pickle.load(handle)

	return info, term2pid


def ExtractSnippt(term, abst, lookahead = 20):
	term = term.lower()
	abst = abst.lower()
	nword = len(term.split(' '))
	tokens = abst.split(" ")  # Split string into a list of tokens
	index = -1
	for i in range(len(tokens)):
		if ' '.join(tokens[i:i+nword]).replace('?','').replace(':','').replace(';','').replace(',','').replace('.','') \
		== term.replace('?','').replace(':','').replace(';','').replace(',','').replace('.',''):
			index = i
			break
	if index==-1:
		return ' '.join(tokens[:min(len(tokens), lookahead*2)])+' ...'
	st = index
	ed = index + nword
	while st > 0 and tokens[st]!='.' and index-st < lookahead:
		st = st - 1
	while ed<len(tokens) and tokens[ed]!='.' and ed - index < lookahead:
		ed = ed + 1
	snippet = ' '.join(tokens[st:ed])
	if st!=0:
		snippet = '... '+snippet
	if ed!=len(tokens):
		snippet = snippet+' ...'
	return snippet

def QueryNodeInfo(t,term2pid, DATA_DIR):
	if t not in term2pid:
		return '','',''
	return ReadAbst(term2pid[t], t, DATA_DIR)

def GetInfoBasedOnID(infos, ind):
	if ind < 0:
		ind = 0
	if ind >= len(infos):
		ind = len(infos) - 1
	ttl, des, url = infos[ind]
	#des = des[0:min(380,len(des))]
	return ttl, des, url


def ScoreAbst(term, title, abst):
	if len(abst.split(' '))<3:
		return 100000
	sc_title = (term.lower() in title.lower()) * 1. / (len(title.split(' '))+1) * 10000
	sc_abst = (term.lower() in abst.lower()) * 1.  / (len(abst.split(' '))+1)
	return (sc_title + sc_abst)*-1

def query_edge(path, DATA_DIR):
	return path

def ReadAbst(pids, term, DATA_DIR):
	pid2sc = {}
	#print (len(pids))=

	abs_dir = DATA_DIR + '/abstract/'

	pids = set(list(pids)[:min(11,len(pids))])
	for file in os.listdir(abs_dir):
		_,st,ed = file.split('_')
		st = int(st)
		ed = int(ed)
		check = False
		for pid in pids:
			if st<=pid and ed>=pid:
				check=True
				break
		#print (file,check)
		if check:
			fin = open(abs_dir+file)
			for line in fin:
				w = line.lower().strip().split('|')
				if int(w[0]) in pids:
					pid2sc['\t'.join([str(pid), w[1], w[2]])] = ScoreAbst(term, w[1], w[2])
			fin.close()
	pid2sc_sorted = sorted(pid2sc.items(), key=operator.itemgetter(1))

	pids = []
	ttls = []
	absts = []
	for i in range(min(10,len(pid2sc_sorted))):
		pid, ttl, abst = pid2sc_sorted[i][0].split('\t')
		#print (pid, ttl, abst,pid2sc_sorted[i][1])
		pids.append(pid)
		ttls.append(ttl)
		absts.append(abst)
	return pids, ttls, absts

def query_node(term, info, term2pid, node2tp, DATA_DIR, ind = 0):
	t1s,t2s,t3s = QueryNodeInfo(term, term2pid, DATA_DIR)
	tmp = []
	for ii in range(len(t3s)):
		t3s[ii] = ExtractSnippt(term, t3s[ii])
		t2s[ii] = ExtractSnippt(term, t2s[ii], lookahead=8)
		if len(t2s[ii])>140:
			t2s[ii] = t2s[ii][:140]
		tmp.append([t2s[ii], t3s[ii],'https://www.ncbi.nlm.nih.gov/pubmed/'+t1s[ii]])
	info[term] = tmp
	#url = info[term][1]
	ttl, des, url = GetInfoBasedOnID(info[term], ind)
	print (ttl, des, url)
	if node2tp[term]=='gene':
		ttl = ttl.upper()
	else:
		ttl = ttl.capitalize()
	if 'ncbi' in url:
		ttl = ttl + ' - NCBI'
	elif 'meshb' in url:
		ttl = ttl + ' - MeSH'
	elif 'genecards' in url:
		ttl = ttl + ' - GeneCards'
	elif 'geneontology' in url:
		ttl = ttl + ' - GO'
	elif 'drugbank' in url:
		ttl = ttl + ' - DrugBank'
	des = list(des)

	for ii in range(len(des)):
		if  (ii+2)<len(des) and des[ii]=='.' and des[ii+1]==' ' and des[ii+2]>='a' and des[ii+2]<='z':
			des[ii+2] = des[ii+2].upper()
	des = ''.join(des)
	return info, ttl, des, url
