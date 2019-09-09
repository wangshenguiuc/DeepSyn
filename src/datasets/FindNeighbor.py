import numpy as np
import collections
import copy
import operator
from scipy import stats
from src.models.network_flow.NetworkFlow import NetworkFlow

class FindNeighbor():

	def __init__(self,s,ImproveNet_obj,stop_word_list=[],exclude_edge_type=[],exclude_edges =set(),max_path_L = 4,include_genes=[],edge_wt_thres = 1,max_ngh=10,max_inf=1000000,max_edge_score = 100000):
		self.s = s
		self.ImproveNet_obj = ImproveNet_obj
		self.stop_word_list = stop_word_list
		self.exclude_edge_type = exclude_edge_type
		self.exclude_edges = exclude_edges
		self.include_genes = include_genes
		self.word_ct = ImproveNet_obj.word_ct
		self.word_type = ImproveNet_obj.word_type
		self.node_set = set()
		self.edge_list = []
		self.pre = collections.defaultdict(dict)

	def get_source(self,n,ngh):
		tp = self.ImproveNet_obj.net[n][ngh].keys()
		return '_'.join(tp)

	def CalNgh(self,G2G_obj,max_layer=5,edge_wt_thres=0.01,max_end_nodes=100,stop_word_list=[],gene_wt_thres=0.01,all_type_same_weight=True,use_direct_gene_edge=False,include_terms=set()):
		self.max_layer = max_layer
		l2nodes = {}
		l2nodes[self.s] = 1
		self.node_set.add(self.s)
		net = self.ImproveNet_obj.net
		g2score = {}
		end_nodes = {}
		self.node2score = {}
		self.node2gene = {}
		self.node2layer = {}
		self.node2score[self.s] = 1
		iter_nodes = set()
		iter_nodes.add(self.s)
		self.node2layer[self.s] = 0
		for l in range(1,max_layer):
			new_iter_nodes = set()

			for n in iter_nodes:
				if n not in net:
					continue
				if all_type_same_weight:
					edge_sum = 0.
				else:
					edge_sum = {}
				valid_ngh = set()
				merge_ngh =  copy.deepcopy(net[n])
				for ngh1 in net[n]:
					for ngh2 in net[n]:
						if ngh2 in merge_ngh and ngh1!=ngh2 and ngh1 in ngh2 and self.word_type[ngh1].split('_')[1]=='disease':
							#print ngh1,ngh2
							merge_ngh.pop(ngh1,None)
							for tp in net[n][ngh1]:
								merge_ngh[ngh2][tp] = max(net[n][ngh1][tp], net[n][ngh2].get(tp,0))

				for ngh in merge_ngh:
					if ngh in stop_word_list:
						continue
					wct1 = self.word_ct['pubmed'].get(n,-1)
					wct2 = self.word_ct['pubmed'].get(ngh,-1)
					if wct1==-1 or wct2==-1:
						continue
					if n+'#'+ngh in self.exclude_edges:
						continue
					if n!=self.s and self.word_type[n].split('_')[1] == self.word_type[ngh].split('_')[1] and self.word_type[ngh].split('_')[1]!='function':
						continue
					if self.word_type[n].split('_')[1] == self.word_type[ngh].split('_')[1] and wct1 < wct2:
						continue
					if [self.word_type[n].split('_')[1], self.word_type[ngh].split('_')[1]] in self.exclude_edge_type:
						continue
					edge_wt = np.sum(merge_ngh[ngh].values())
					if edge_wt < edge_wt_thres:
						continue
					wt = self.word_type[ngh].split('_')[1]
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
					edge_wt = np.sum(merge_ngh[ngh].values())
					wt = self.word_type[ngh].split('_')[1]
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
					if n in end_nodes:
						end_nodes.pop(n, None)
			iter_nodes = set(new_iter_nodes)
			#print l, len(l2nodes),len(iter_nodes)
		ngene = len(G2G_obj.g2i)
		gvec = np.zeros(ngene)
		visit_g = 0
		G2G_network = G2G_obj.sparse_network.toarray()
		sorted_x = sorted(end_nodes.items(), key=operator.itemgetter(1))
		sorted_x.reverse()
		for i,ni in enumerate(sorted_x):
			#if n not in l2nodes[max_layer-1]:
			#	continue
			n = sorted_x[i][0]
			if i > max_end_nodes and n not in include_terms:
				continue
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
				if n not in self.node2gene:
					self.node2gene[n] = collections.defaultdict(dict)
				for j in range(ngene):
					self.node2gene[n][G2G_obj.i2g[j].lower()][g] = gec_l[j]
				visit_g += 1
		self.g2score = {}
		for i in G2G_obj.i2g:
			self.g2score[G2G_obj.i2g[i].lower()] = gvec[i]
		#for e in self.edge_list:
		#	print e,self.word_type[e[0]],self.word_type[e[1]]
		#print self.edge_list
		#for n in self.node2score:
		#	if self.word_type[n].split('_')[1]!='gene':
		#		print n,self.node2score[n]
		self.l2nodes = l2nodes
		return self.g2score,gvec

	def GetPathway(self, G2G_obj, pv_sign=0.05):
		npath = len(self.node2gene)
		ngene = len(G2G_obj.i2g)
		p2g = np.zeros((npath,ngene))
		for i,path in enumerate(self.node2gene):
			for g in range(ngene):
				p2g[i][g] = np.sum(self.node2gene[path][G2G_obj.i2g[g].lower()].values())
			p2g[i,:] /= np.sum(p2g[i,:])
		self.path2gene = {}
		for i,path in enumerate(self.node2gene):
			self.path2gene[path] = {}
			gvec = p2g[i,:] - np.mean(p2g,axis=0)
			cand_gene = np.where(gvec>0)[0]
			for g in cand_gene:
				pv = stats.ttest_1samp(p2g[:,g],p2g[i,g])[1]
				if pv * len(cand_gene) < pv_sign:
					self.path2gene[path][G2G_obj.i2g[g]] = pv* len(cand_gene)
		return self.path2gene

	def CalQuality(self, node_set, edge_list, nnode_min = 4, nnode_max = 15, nlayer_min = 3,nfunc_min = 1):
		s = self.s
		ntype = {}
		for n in node_set:
			nt = self.word_type[n].split('_')[1]
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
		nlayer = max(n2l.values())
		nnode = len(node_set)
		confidence = nfunc + nlayer
		#if nnode < nnode_min or nnode > nnode_max or nlayer < nlayer_min or nfunc < nfunc_min:
		#	confidence = 0
		detailed_confidence = [nfunc,nlayer,ngene,ngene_ngh]
		return confidence,detailed_confidence

	def GetSubNetwork(self, gset, cutoff = 1e-9, topk = 5, include_terms = set(),stop_word_list=set()):
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
			if g not in self.g2score:
				continue
			node_weight[g] = self.g2score[g]
			node_weight[self.s] = 1
			n2score = {}
			for n in self.node2gene:
				if n in stop_word_list:
					continue
				#maxi = np.argmax(self.node2gene[n][g].values())
				#print maxi,len(self.node2gene[n][g].values())
				#if not np.isscalar(maxi):
				#	maxi = maxi[0]
				n2score[n] = np.sum(self.node2gene[n][g].values())#np.max(self.node2gene[n][g].values())
				#n2inter[n] = self.node2gene[n][g].keys()[maxi]
				sum += n2score[n]
			sorted_x = sorted(n2score.items(), key=operator.itemgetter(1))
			sorted_x.reverse()

			for i in range(min(topk,len(sorted_x))):
				n,sc = sorted_x[i]
				if sc < cutoff:
					break
				if n not in ngh2gene:
					ngh2gene[n] = {}
				ngh2gene[n][g] = sc

		last_layer_node_set,last_layer_node_weight,last_layer_edge_list, select_node = self.get_cover_set(ngh2gene)
		#print select_node
		self.nselect_node = len(select_node)
		node_set = node_set.union(last_layer_node_set)
		for n in last_layer_node_weight:
			node_weight[n] = last_layer_node_weight[n]
		for e in last_layer_edge_list:
			edge_list.append(e)
		ct = 0
		while len(select_node)>0:
			new_node = set()
			for n in select_node:
				if n==self.s:
					continue
				for pre in self.pre[n]:
					if pre in stop_word_list:
						continue
					if pre not in node_set:
						new_node.add(pre)
					node_set.add(pre)
					node_weight[pre] = self.node2score[pre]
					edge_list.append([pre, n, self.pre[n][pre], self.get_source(pre,n)])
			select_node = new_node
		return node_set, edge_list,node_weight
		'''
		while len(select_node)>0 and ct<self.max_layer:
			new_node = set()
			for n in select_node:
				if n==self.s:
					continue
				for pre in self.pre[n]:
					if pre not in node_set:
						node_set.add(pre)
						new_node.add(pre)
						node_weight[pre] = self.node2score[pre]
						edge_list.append([pre, n, self.pre[n][pre], self.get_source(pre,n)])
			select_node = new_node
			ct += 1
		'''
		return node_set, edge_list,node_weight

	def dfs_net(self,layer,cur_node,net,gset,net_pre,include_terms):
		if len(include_terms)==0:
			return [],include_terms
		if cur_node in gset:
			path = [cur_node]
			p = net_pre[cur_node]
			while p!=self.s:
				path.append(p)
				p = net_pre[p]
			path.append(p)
			Find = False
			for p in path:
				if p in include_terms:
					Find = True
					include_terms.remove(p)
			if Find:
				return [path], include_terms
			else:
				return [], include_terms

		if cur_node not in net or layer>self.max_layer+5:
			return [],include_terms
		cur_ans = []
		ngh_d = {}
		for n in net[cur_node]:
			ngh_d[n] = net[cur_node][n]
		ngh_d = sorted(ngh_d.items(), key=operator.itemgetter(1))
		ngh_d.reverse()
		for ni in ngh_d:
			n = ni[0]
			#print n, ni[1]
			net_pre[n] = cur_node
			ans,include_terms = self.dfs_net(layer+1,n,net,gset,net_pre,include_terms)
			for a in ans:
				cur_ans.append(a)
		return cur_ans,include_terms

	def bfs_net(self,net,st_nodes,end_nodes):
		pre = {}
		dis = {}
		step = 0
		visit_nodes = set()
		for s in st_nodes:
			visit_nodes.add(s)
		end_nodes_copy = copy.deepcopy(end_nodes)
		while len(end_nodes_copy) > 0 and len(st_nodes) > 0:
			new_st_nodes = set()
			for s in st_nodes:
				dis[s] = step
				for n in net[s]:
					if n in dis:
						continue
					pre[n] = s
					new_st_nodes.add(n)
					if n in end_nodes_copy:
						end_nodes_copy.remove(n)
			st_nodes = new_st_nodes
			#print new_st_nodes
			step += 1
		return pre,dis

	def PruneSubNetwork(self, gene_set, node_set, edge_list, include_terms=set()):
		gset = gene_set
		st_nodes = set([self.s])
		end_nodes = gset
		mid_nodes = include_terms

		net = collections.defaultdict(dict)
		for e in edge_list:
			e1,e2,wt = e[:3]
			net[e1][e2] = wt

		st_mid_pre,st_mid_dis = self.bfs_net(net,st_nodes,mid_nodes)

		#print 'st',st_mid_pre
		#print 'mid',mid_end_pre
		filter_node_set = set()
		tmp_edge_list = []
		for n in mid_nodes:
			p = n
			while p not in st_nodes and p in st_mid_pre:
				filter_node_set.add(p)
				filter_node_set.add(st_mid_pre[p])
				tmp_edge_list.append([st_mid_pre[p], p])
				#print p, st_mid_pre[p]
				p = st_mid_pre[p]

		for t in include_terms:
			mid_nodes = set([t])
			mid_end_pre,mid_end_dis = self.bfs_net(net,mid_nodes,end_nodes)
			for n in end_nodes:
				p = n
				while p not in mid_nodes and p in mid_end_pre:
					filter_node_set.add(p)
					filter_node_set.add(mid_end_pre[p])
					tmp_edge_list.append([mid_end_pre[p], p])
					#print p, mid_end_pre[p]
					p = mid_end_pre[p]


		filter_edge_list = []
		for e in edge_list:
			if e[0] in filter_node_set and e[1] in filter_node_set:
				filter_edge_list.append(e)
		#print filter_edge_list
		return filter_node_set,filter_edge_list

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
