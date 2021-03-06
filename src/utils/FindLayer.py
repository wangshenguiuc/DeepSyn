import numpy as np
import collections
import copy
import operator
from scipy import stats
import os

def get_pvalue_path(DATA_DIR, function):
	pvalue_dir = DATA_DIR + '/data/pvalue/preprocess/' + function[0:2] + '/'
	if not os.path.exists(pvalue_dir):
		os.makedirs(pvalue_dir)
	return pvalue_dir + function

class FindLayer():

	def __init__(self,st_nodes,ed_nodes,ImproveNet_obj,kw_st, kw_ed,
	stop_word_list=[],exclude_edge_type=[],net_topk=2,exclude_edges =set(),
	max_layer = 4,include_genes=[],edge_wt_thres = 1,max_end_nodes=100,prop = True, DATA_DIR=''):
		self.source = st_nodes
		self.target = ed_nodes
		self.source_type = kw_st
		self.target_type = kw_ed
		self.ImproveNet_obj = ImproveNet_obj
		self.stop_word_list = stop_word_list
		self.exclude_edge_type = exclude_edge_type
		self.exclude_edges = exclude_edges
		self.include_genes = include_genes
		self.word_ct = ImproveNet_obj.word_ct
		self.word_type = ImproveNet_obj.word_type
		self.node_set = set()
		self.net_topk = net_topk
		self.edge_list = []
		self.max_end_nodes = max_end_nodes
		self.max_layer = max_layer
		self.edge_wt_thres = edge_wt_thres
		self.DATA_DIR = DATA_DIR
		self.pre = collections.defaultdict(dict)
		if self.target_type=='gene' and prop==True:
			self.prop = True
		else:
			self.prop = False



	def get_source(self,n,ngh):
		tp = self.ImproveNet_obj.net[n][ngh].keys()
		return '_'.join(tp)

	def get_type(self,w):
		if w not in self.word_type:
			return 'NoneType'
		tp = self.word_type[w].split('_')[1]
		return tp

	def CalNgh(self,G2G_obj,all_type_same_weight=True,use_direct_gene_edge=False,include_terms=set()):
		l2nodes = {}
		g2score = {}
		end_nodes = {}
		self.node2score = {}
		self.node2layer = {}
		iter_nodes = set()
		net = self.ImproveNet_obj.net

		for s in self.source:
			l2nodes[s] = 1
			self.node_set.add(s)
			self.node2score[s] = 1
			iter_nodes.add(s)
			self.node2layer[s] = 0
		for l in range(1,self.max_layer):
			new_iter_nodes = set()
			#print l,self.max_layer,len(end_nodes),len(iter_nodes)
			for n in iter_nodes:
				#print n
				if n not in self.ImproveNet_obj.net:
					continue
				if all_type_same_weight:
					edge_sum = 0.
				else:
					edge_sum = {}
				valid_ngh = set()
				merge_ngh =  copy.deepcopy(self.ImproveNet_obj.net[n])
				#for ngh1 in self.ImproveNet_obj.net[n]:
				#	for ngh2 in self.ImproveNet_obj.net[n]:
				#		if ngh2 in merge_ngh and ngh1!=ngh2 and ngh1 in ngh2 and self.word_type[ngh1].split('_')[1]=='disease':
				#			merge_ngh.pop(ngh1,None)
				#			for tp in self.ImproveNet_obj.net[n][ngh1]:
				#				merge_ngh[ngh2][tp] = max(self.ImproveNet_obj.net[n][ngh1][tp], self.ImproveNet_obj.net[n][ngh2].get(tp,0))
				for ngh in merge_ngh:
					if ngh in self.stop_word_list:
						continue

					wct1 = self.word_ct['pubmed'].get(n,-1)
					wct2 = self.word_ct['pubmed'].get(ngh,-1)
					#print n,ngh,wct1,wct2
					#print 'd',ngh,wct1,wct2
					#if wct1==-1 or wct2==-1:
					#	continue
					if n+'#'+ngh in self.exclude_edges:
						continue
					#if n!=self.source and self.get_type(n) == self.get_type(ngh) and self.get_type(ngh)!='function':
					#	continue
					if self.get_type(n) == self.get_type(ngh) and wct1 < wct2 and self.get_type(ngh)!='gene':
						continue
					if [self.get_type(n), self.get_type(ngh)] in self.exclude_edge_type:
						continue
					if self.get_type(n) == self.get_type(ngh) and self.get_type(ngh)=='gene':
						if n in self.pre:
							old_ngh = list(self.pre[n].keys())[0]
							if self.word_type[old_ngh].split('_')[1] == 'gene':
								continue
					edge_wt = np.sum(np.array(list(merge_ngh[ngh].values())))
					if edge_wt < self.edge_wt_thres :
						continue
					wt = self.get_type(ngh)
					if wt == 'gene' and not use_direct_gene_edge:
						continue
					if wt == 'gene' and len(self.include_genes) > 0 and ngh not in self.include_genes:
						continue
					if all_type_same_weight:
						edge_sum += edge_wt
					else:
						edge_sum[wt] = edge_sum.get(wt,0.) + edge_wt

					valid_ngh.add(ngh)
				for ngh in valid_ngh:
					edge_wt = np.sum(np.array(list(merge_ngh[ngh].values())))
					wt = self.get_type(ngh)
					if all_type_same_weight:
						new_edge_wt = edge_wt / edge_sum
					else:
						new_edge_wt = edge_wt / edge_sum[wt]
					new_node_wt = new_edge_wt*l2nodes[n]
					l2nodes[ngh] = l2nodes.get(ngh,0) + new_node_wt
					self.pre[ngh][n] = l
					self.node2layer[ngh] = l
					self.node_set.add(ngh)
					self.edge_list.append([n, ngh, new_edge_wt, self.get_source(n,ngh)])
					end_nodes[ngh] = self.node2score[n]
					new_iter_nodes.add(ngh)
					self.node2score[ngh] = l2nodes[ngh]
					#if n in end_nodes:
					#	end_nodes.pop(n, None)
				#print 'end_nodes',end_nodes
			iter_nodes = set(new_iter_nodes)

		#print 'end_nodes',end_nodes
		#print  self.target,self.source,self.target_type
		#sys.exit(-1)
		tgt2weight,endnode2tgt = self.PropEndNgh(end_nodes,G2G_obj=G2G_obj,include_terms=include_terms,prop=self.prop)

		node_set, edge_list, node_weight = self.GetSubNetwork(tgt2weight,endnode2tgt,topk=self.net_topk)
		#for n in node_set:
		#	if self.get_type(n)  == 'drug':
		#		print n,'aft'
		#node2pvalue = self.CalPvalue(node_set, G2G_obj)
		return node_set, edge_list, node_weight

	def ReadFunc2GeneData(self,gene_set,G2G_obj,function_set, pvalue_dec=1e6,function_score_file=None,background_pvalue_file=None):
		if function_score_file is None:
			function_score_file = self.DATA_DIR + '/data/pvalue/function_score/phrase/2_0.01/all_new.txt'
		f2g_sc =  collections.defaultdict(dict)
		for d in function_set:
			fpath = get_pvalue_path(self.DATA_DIR, d)
			if not os.path.isfile(fpath):
				continue
			fin = open(fpath)
			line = fin.readline()
			w = line.strip().split('\t')
			for g in gene_set:
				if g.upper() not in G2G_obj.g2i:
					continue
				i = G2G_obj.g2i[g.upper()]
				f2g_sc[d][g] = float(w[i])
			fin.close()

		if background_pvalue_file is None:
			background_pvalue_file = self.DATA_DIR + '/data/pvalue/function_score/pvalue/' + str(int(pvalue_dec))
		fin = open(background_pvalue_file)
		background_sc = []
		for line in fin:
			w = line.strip()
			background_sc.append(float(w))
		fin.close()
		background_sc = np.array(background_sc)
		return f2g_sc,background_sc

	def CalPvalue(self,node_set, G2G_obj,stop_word = []):
		gene_set = set()
		function_set = set()
		for n in node_set:
			if self.get_type(n)=='gene':
				gene_set.add(n)
			if self.get_type(n)=='function':
				function_set.add(n)
		f2g_sc,background_sc = self.ReadFunc2GeneData(gene_set, G2G_obj, function_set)
		node2pvalue = {}
		for f in function_set:
			if f in stop_word:
				continue
			if f not in f2g_sc:
				continue
			sc = 0.
			ngene_term = 0
			for g in gene_set:
				if g in f2g_sc[f]:
					sc+= f2g_sc[f][g]
					ngene_term+=1
			if ngene_term==0:
				continue
			sc /= ngene_term
			pv = (len(np.where(background_sc>sc)[0]) + 1) * 1. / len(background_sc)
			node2pvalue[f] = pv
		for f in node_set:
			if f not in node2pvalue:
				node2pvalue[f] = 1.
		return node2pvalue


	def PropEndNgh(self, end_nodes, G2G_obj={}, include_terms=set(),prop=True):
		net = self.ImproveNet_obj.net
		endnode2target = {}
		tgt2score = {}
		ngene = len(G2G_obj.g2i)
		gvec = np.zeros(ngene)
		visit_g = 0
		G2G_network = G2G_obj.sparse_network.toarray()
		sorted_x = sorted(end_nodes.items(), key=operator.itemgetter(1))
		sorted_x.reverse()
		#print ngene
		for i,ni in enumerate(sorted_x):
			n = sorted_x[i][0]
			if i > self.max_end_nodes and n not in include_terms:
				continue
			if prop:
				gsum = 0.
				if n.upper() in G2G_obj.g2i:
					if n not in net:
						net[n] = {}
					net[n][n] = 1
				if n not in net:
					continue
				for g in net[n]:
					if g.upper() not in G2G_obj.g2i:
						continue
					gsum += 1.
				for g in net[n]:
					if g.upper() not in G2G_obj.g2i:
						continue
					wt = 1. / gsum
					gid = G2G_obj.g2i[g.upper()]
					#print wt, g, n
					gec_l = self.node2score[n] * G2G_obj.rwr[gid,:] * wt
					gvec += gec_l
					if n not in endnode2target:
						endnode2target[n] = collections.defaultdict(dict)
					for j in range(ngene):
						endnode2target[n][G2G_obj.i2g[j].lower()][g] = gec_l[j]
					visit_g += 1
			else:
				for t in self.target:
					if n in net and t in net[n]:
						if n not in endnode2target:
							endnode2target[n] = collections.defaultdict(dict)
						endnode2target[n][t][t] = self.node2score[n]
					if n==t:
						if n not in endnode2target:
							endnode2target[n] = collections.defaultdict(dict)
						endnode2target[n][t][t] = self.node2score[n]
		if prop:
			for i in G2G_obj.i2g:
				tgt2score[G2G_obj.i2g[i].lower()] = gvec[i]
			for t in self.target:
				tgt2score[t] = 1.
		else:
			for t in self.target:
				tgt2score[t] = 1.
		#print tgt2score
		#print 'endnode2target',endnode2target
		return tgt2score,endnode2target



	def CalQuality(self, node_set, edge_list, nnode_min = 4, nnode_max = 15, nlayer_min = 3,nfunc_min = 1):
		s = self.source
		ntype = {}
		for n in node_set:
			nt = self.get_type(n)
			ntype[nt] = ntype.get(nt,0)+1
		nfunc = ntype.get('function',0)
		ngene =  ntype.get('gene',0)
		nnode_type = len(ntype)
		nsource = []
		net = collections.defaultdict(dict)
		gene_ngh = set()
		for e in edge_list:
			e1,e2 = e[:2]
			net[e1][e2] = 1
			if self.word_type[e1].split('_')[1] == 'gene' and self.word_type[e2].split('_')[1] == 'function':
				gene_ngh.add(e2)
			if self.word_type[e1].split('_')[1] == 'function' and self.word_type[e2].split('_')[1] == 'gene':
				gene_ngh.add(e1)
		ngene_ngh = len(gene_ngh)
		n2l = {}
		n2l[s] = 0
		node_set = set()
		node_set.add(s)
		while len(node_set) > 0:
			new_node_set = set()
			for n in node_set:
				for ngh in net[n]:
					if ngh not in n2l:
						new_node_set.add(ngh)
					n2l[ngh] = n2l[n]+1
			node_set = new_node_set
			#print len(node_set)
		#print net,n2l
		nsource = np.mean(nsource)
		nlayer = max(list(n2l.values()))
		nnode = len(node_set)
		confidence = nfunc + nlayer
		#if nnode < nnode_min or nnode > nnode_max or nlayer < nlayer_min or nfunc < nfunc_min:
		#	confidence = 0
		detailed_confidence = [nfunc,nlayer,ngene,ngene_ngh]
		return confidence,detailed_confidence

	def GetSubNetwork(self, tgt2weight,endnode2tgt, cutoff = 0, topk = 5, include_terms = set(),stop_word_list=set()):
		#print 'here',self.tgt2score,self.target
		gset = self.target
		node_set = set()
		node_weight = {}
		edge_list = []
		n2inter = {}
		sum = 0.
		if len(gset) == 1:
			gset_label = list(gset)[0]
		else:
			gset_label = 'gene set'
		select_node = set()
		ngh2gene = {}
		for g in gset:
			if g not in tgt2weight:
				continue
			node_weight[g] = tgt2weight[g]
			for s in self.source:
				node_weight[s] = 1
			n2score = {}
			for n in endnode2tgt:
				if n in stop_word_list:
					continue
				#maxi = np.argmax(self.node2gene[n][g].values())
				#print maxi,len(self.node2gene[n][g].values())
				#if not np.isscalar(maxi):
				#	maxi = maxi[0]
				n2score[n] = np.sum(list(endnode2tgt[n][g].values()))#np.max(self.node2gene[n][g].values())
				#n2inter[n] = self.node2gene[n][g].keys()[maxi]\
				sum += n2score[n]
			sorted_x = sorted(n2score.items(), key=operator.itemgetter(1))
			sorted_x.reverse()
			for i in range(min(topk,len(sorted_x))):
				n,sc = sorted_x[i]
				if sc < cutoff:
					break
				if n not in ngh2gene:
					ngh2gene[n] = {}
				#if n==g:
				#	continue
				ngh2gene[n][g] = sc
		last_layer_node_set,last_layer_node_weight,last_layer_edge_list, select_node = self.get_cover_set(ngh2gene)
		#print select_node
		self.nselect_node = len(select_node)
		node_set = node_set.union(last_layer_node_set)

		for n in last_layer_node_weight:
			node_weight[n] = last_layer_node_weight[n]
		for e in last_layer_edge_list:
			pre,n,wt,type = e
			if pre==n:
				continue
			edge_list.append(e)
		ct = 0
		while len(select_node)>0:
			new_node = set()
			for n in select_node:
				if n in self.source:
					continue
				for pre in self.pre[n]:
					if pre in stop_word_list:
						continue
					if pre not in node_set:
						new_node.add(pre)
					node_set.add(pre)
					node_weight[pre] = self.node2score[pre]
					if n==pre:
						continue
					edge_list.append([pre, n, self.pre[n][pre], self.get_source(pre,n)])
			select_node = new_node
		return node_set, edge_list,node_weight


	def get_cover_set(self, ngh2gene):
		ngh2gene_old = copy.deepcopy(ngh2gene)
		remain_gene = set()
		for n in ngh2gene:
			for g in ngh2gene[n]:
				remain_gene.add(g)
		select_node = set()
		while len(remain_gene) > 0:
			maxn = 0
			maxv = 0
			for n in ngh2gene:
				ct = 0.
				for g in ngh2gene[n]:
					if g in remain_gene:
						ct+=1
				if ct > maxv:
					maxv = len(ngh2gene[n])
					maxn = n
			select_node.add(maxn)
			for g in ngh2gene[maxn]:
				if g in remain_gene:
					remain_gene.remove(g)
			del ngh2gene[maxn]
		node_set = set()
		node_weight = {}
		edge_list = []
		for n in select_node:
			for g in ngh2gene_old[n]:
				sc = ngh2gene_old[n][g]
				node_set.add(g)
				node_weight[g] = node_weight.get(g,0) + sc
				node_set.add(n)
				node_weight[n] = self.node2score[n]
				edge_list.append([n, g, sc, 'PPI'])
		#print len(ngh2gene_old),'in',len(select_node)
		return node_set,node_weight,edge_list,select_node
