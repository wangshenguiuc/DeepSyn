import numpy as np
import collections
import copy
import operator
from src.models.network_flow.NetworkFlow import NetworkFlow

class SubGraph():

	def __init__(self,ImproveNet_obj,SKhop,TKhop,s,tgt_set,stop_word_list=[],max_path_L = 4,edge_wt_thres = 1,max_ngh=10,max_inf=1000000,max_edge_score = 100000):
		self.dis2source = SKhop.dis2source
		self.dis2target = TKhop.dis2source
		self.s = s
		self.tgt_set = tgt_set
		self.word_type = ImproveNet_obj.word_type
		self.bp_netflow = collections.defaultdict(dict)
		self.netflow = collections.defaultdict(dict)
		self.graph_edge_type = ImproveNet_obj.graph_edge_type
		self.word_ct = ImproveNet_obj.word_ct


		seed = tgt_set
		#for t in seed:
		#	self.get_K_ngh(t,ImproveNet_obj.bp_net,ImproveNet_obj.net,max_path_L,edge_wt_thres,'from_target')
		self.graph_node_set = set()
		#self.graph_node_set.add(s)
		for si in self.dis2source[s]:
			if si in stop_word_list:
				continue
			dis2s = self.dis2source[s].get(si,max_inf)
			dis2t = max_inf
			for t in tgt_set:
				dis2t = min(dis2t,self.dis2target[t].get(si,max_inf))
			if dis2s + dis2t <= max_path_L:
				self.graph_node_set.add(si)


		#self.graph_node_set = self.graph_node_set.union(seed)
		if len(self.graph_node_set) > 0:
			self.graph_node_set.add('@super target')
		for t1 in self.graph_node_set:
			new_d = {}
			if t1 not in ImproveNet_obj.bp_net:
				continue
			for t2 in ImproveNet_obj.bp_net[t1]:
				if t2 not in ImproveNet_obj.bp_net[t1]:
					continue
				for source in ImproveNet_obj.bp_net[t1][t2]:
					#if t1==source_node:
					#	print t1,t2,select_edge_ct[t1][t2],edge_thres
					if ImproveNet_obj.bp_net[t1][t2][source] > edge_wt_thres:
						new_d[t2] = new_d.get(t2,0) + ImproveNet_obj.bp_net[t1][t2][source]
			sort_x = sorted(new_d.items(),key=operator.itemgetter(1))
			sort_x.reverse()

			for i in range(len(sort_x)):
				t2,sc = sort_x[i]
				if i>max_ngh and sc<max_edge_score:
					continue
				self.bp_netflow[t1][t2] = sc
				self.netflow[t2][t1] = sc
			'''
			for i in range(min(len(sort_x),max_ngh)):
				t2,sc = sort_x[i]
				#if i>max_ngh and sc<100000:
				#	continue
				self.bp_netflow[t1][t2] = sc
				self.netflow[t2][t1] = sc
			'''
		#for t in tgt_set:
		#	self.netflow[t]['@super target'] = max_inf
		#	self.bp_netflow['@super target'][t] = max_inf
		self.node2i = {}
		self.i2node = {}
		for s in self.graph_node_set:
			nnode = len(self.node2i)
			self.i2node[nnode] = s
			self.node2i[s] = nnode


	def get_K_ngh(self,s,net,ref_net,max_path_L,edge_thres,mode):
		if mode=='from_source':
			if s in self.dis2source:
				return
			cur_D = self.dis2source
			self.dis2source[s][s] = 0
		elif mode == 'from_target':
			if s in self.dis2target:
				return
			cur_D = self.dis2target
			self.dis2target[s][s] = 0
		else:
			raise ValueError('wrong mode type')
		visited = set()
		visited.add(s)
		for iter in range(max_path_L):
			new_node = set()
			for ss in visited:
				if ss not in net:
					continue
				for e in net[ss]:
					if e in cur_D[s]:
						continue
					if mode =='from_target' and ref_net[e][ss] < edge_thres:
						#print e,ss
						continue
					new_node.add(e)
			visited = new_node
			for ss in visited:
				if ss not in cur_D[s]:
					if mode=='from_source':
						self.dis2source[s][ss] = iter+1
					elif mode == 'from_target':
						self.dis2target[s][ss] = iter+1
					else:
						raise ValueError('wrong mode type')
		return


	def GraphScoreDFS(self,cur_node,cur_depth,path,global_all_path,edge_type_ct):
		for etype in edge_type_ct:
			#if etype[0] == 'tissue' or etype[1] == 'tissue':
			#	return []
			if etype == ('gene','gene') and edge_type_ct[etype] > 0:#self.max_dup_edge_type:
				return []
			if etype == ('function','function') and edge_type_ct[etype] > self.max_dup_edge_type+1:
				return []
			if etype == ('disease','disease') and edge_type_ct[etype] > 0:
				return []
		if cur_node in self.tgt_set:
			#print path
			global_all_path.append(path)
			return global_all_path
		if cur_depth == self.dfs_max_depth:
			#print path
			return []
		argu_path = []
		for ngh in self.netflow[cur_node]:
			if self.use_edge_type_filter and [self.word_type[cur_node].split('_')[1],self.word_type[ngh].split('_')[1]] not in self.graph_edge_type:
				continue
			new_edge_type_ct = copy.deepcopy(edge_type_ct)
			etype = (self.word_type[cur_node].split('_')[1],self.word_type[ngh].split('_')[1])
			wct1 = self.word_ct['pubmed'].get(cur_node,-1)
			wct2 = self.word_ct['pubmed'].get(ngh,-1)
			if etype[0] == etype[1] and wct1>0 and wct2>0 and wct1 < wct2/2:
				continue
			new_edge_type_ct[etype] = new_edge_type_ct.get(etype,0) + 1
			new_path = self.GraphScoreDFS(ngh,cur_depth+1,path+'\t'+ngh,global_all_path,new_edge_type_ct)
			#if len(new_path) > 0:
			#	argu_path.append(new_path)
		return global_all_path

	def CalSubGraphScore(self, ImproveNet_obj,G2G_obj, image_file=None,dfs_max_depth = 5,use_edge_type_filter=False,max_dup_edge_type=2,tgt_baseline_score={}):
		self.max_dup_edge_type = max_dup_edge_type
		self.use_edge_type_filter = use_edge_type_filter
		self.dfs_max_depth = dfs_max_depth
		edge_type_ct = {}
		all_path = self.GraphScoreDFS(self.s,0,self.s,[],edge_type_ct)
		npath = len(all_path)
		edge_d = {}
		node_set = set()
		multiplier = 1000
		for p in range(npath):
			path = all_path[p].split('\t')
			for w in range(0,len(path)-1):
				node_set.add(path[w])
				node_set.add(path[w+1])
				edge_d[(path[w],path[w+1])] = edge_d.get((path[w],path[w+1]),0) + 1
		find_tgt = self.tgt_set.intersection(node_set)
		#print len(find_tgt)
		final_netflow_edge = collections.defaultdict(dict)
		subgraph_edge = collections.defaultdict(dict)
		final_netflow_node = set()
		for k in edge_d:
			final_netflow_edge[k[0]][k[1]] = self.netflow[k[0]][k[1]]*multiplier
			#e1,e2 = min(k[0],k[1]),max(k[0],k[1])
			e1,e2 = k[0],k[1]
			subgraph_edge[(e1,e2)] = 1
			final_netflow_node.add(k[0])
			final_netflow_node.add(k[1])
		for tgt in self.tgt_set:
			if tgt not in self.node2i:
				print tgt,'not found'
				continue
			final_netflow_edge[tgt]['@super target'] = 100000
			final_netflow_edge['@super target'][tgt] = 100000
		final_netflow_node.add('@super target')

		edge_list = []
		node_list  =set()
		for k in edge_d:
			e1 = k[0]
			e2 = k[1]
			type_set = set()
			if e1 in ImproveNet_obj.net and e2 in ImproveNet_obj.net[e1]:
				for k in ImproveNet_obj.net[e1][e2].keys():
					type_set.add(k)
			if e2 in ImproveNet_obj.net and e1 in ImproveNet_obj.net[e2]:
				for k in ImproveNet_obj.net[e2][e1].keys():
					type_set.add(k)
			type_list = list(type_set)
			edge_list.append([e1,e2,self.netflow[e1][e2]*multiplier,self.netflow[e1][e2]*multiplier,type_list])
			node_list.add(e1)
			node_list.add(e2)

		net_flow_obj = NetworkFlow(self.s,'@super target',final_netflow_node,final_netflow_edge,self.node2i,self.i2node,ImproveNet_obj.net)
		##print net_flow_obj.edge_list
		net_score = -1
		npath = 0
		for e in net_flow_obj.edge_list:
			e1,e2,w,mw,type_list = e
			if e1=='@super target' or e2=='@super target':
				net_score = float(w)
				npath += 1
		'''
		ngene = len(G2G_obj.g2i)
		gvec = np.zeros(ngene)
		for e in net_flow_obj.edge_list:
			e1,e2,w,mw,type_list = e
			if e1 not in self.tgt_set and e2 in self.tgt_set:
				continue
			if e2 in self.tgt_set:
				e1,e2 = e2,e1
			for g in self.netflow[e2]:
				if g not in G2G_obj.g2i:
					continue
				gid = G2G_obj.g2i[g]
				gvec += w * G2G_obj.rwr[gid,:]
		'''
		net_confidence,detailed_confidence = self.CalSubGraphConfidence(ImproveNet_obj, net_flow_obj)
		if image_file and net_score!=-1:
			conf_str = "%02d"%net_confidence
			#net_flow_obj.plot_subgraph(self.word_type,find_tgt,image_file[0]+conf_str+image_file[1],subgraph_edge,final_netflow_node,tgt_baseline_score)
			net_flow_obj.plot_network_flow(self.s,self.word_ct, net_flow_obj.node_list, net_flow_obj.edge_list, self.word_type,find_tgt,image_file[0]+image_file[1],final_netflow_edge,tgt_baseline_score,multiplier)

		return net_score,net_confidence,detailed_confidence,npath,net_flow_obj.edge_list,net_flow_obj.node_list,len(find_tgt)

	def CalSubGraphConfidence(self, ImproveNet_obj, net_flow_obj, nnode_min = 4, nnode_max = 15, nlayer_min = 3,nfunc_min = 1):
		s = self.s
		tgt = self.tgt_set
		ntype = {}
		for n in net_flow_obj.node_list:
			if n not in self.word_type:
				continue
			nt = self.word_type[n].split('_')[1]
			ntype[nt] = ntype.get(nt,0)+1
		nfunc = ntype.get('function',0)/(ntype.get('gene',0)+1)
		nnode_type = len(ntype)
		nlayer = len(ntype)
		nsource = []
		for e in net_flow_obj.edge_list:
			e1,e2,w,mw,type_list = e
			ns1 = 0
			ns2 = 0
			ns = len(type_list)
			if ns==0:
				continue
			nsource.append(ns)
		nsource = np.mean(nsource)
		nnode = len(net_flow_obj.node_list)
		confidence = nfunc + nlayer + nnode_type
		if nnode < nnode_min or nnode > nnode_max or nlayer < nlayer_min or nfunc < nfunc_min:
			confidence = 0
		detailed_confidence = [nnode,nfunc,nlayer,nsource]
		return confidence,detailed_confidence
