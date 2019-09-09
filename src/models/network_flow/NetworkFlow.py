from ortools.graph import pywrapgraph
from graphviz import Digraph

class NetworkFlow():
	def run_network_flow(self,s,t,node_set,net,node2i,i2node,whole_network):
		self.edge_list = []
		self.node_list = set()
		self.s = s
		self.t = t
		if s not in node2i:
			return [],[]
		nedge = -1
		for edge_mult in [1]:
			max_flow = pywrapgraph.SimpleMaxFlow()
			for g1 in node_set:
				for g2 in net[g1]:
					#if g2 not in select_node_set:
					#    continue
					#print node2i[g1],node2i[g2],net_flow_edge[g1][g2]
					max_flow.AddArcWithCapacity(node2i[g1],node2i[g2],int(net[g1][g2]*edge_mult))
					#print node2i[g1],node2i[g2],int(net_flow_edge[g1][g2]*edge_mult)

			if max_flow.Solve(node2i[s],node2i['@super target']) == max_flow.OPTIMAL:
				nedge = 0
				self.max_flow_score = int(max_flow.OptimalFlow())
				for i in range(max_flow.NumArcs()):
					if max_flow.Flow(i)>0:
						nedge += 1
			else:
				print('There was an issue with the max flow input.')
			if nedge>1:
				for i in range(max_flow.NumArcs()):
					if max_flow.Flow(i)>0:
						e1 = i2node[max_flow.Tail(i)]
						e2 = i2node[max_flow.Head(i)]
						type_set = set()
						if e1 in whole_network and e2 in whole_network[e1]:
							for k in whole_network[e1][e2].keys():
								type_set.add(k)
						if e2 in whole_network and e1 in whole_network[e2]:
							for k in whole_network[e2][e1].keys():
								type_set.add(k)
						type_list = list(type_set)
						self.edge_list.append([e1,e2,max_flow.Flow(i),max_flow.Capacity(i),type_list])
						self.node_list.add(i2node[max_flow.Tail(i)])
						self.node_list.add(i2node[max_flow.Head(i)])
				break
		#return self.edge_list,self.node_list
	def plot_subgraph(self,s,word_type,tgt_set,output_file,netflow_edge,netflow_node,tgt_baseline_score):
		dot = Digraph(comment='Network flow')
		dot.format = 'eps'

		for node in netflow_node:
			if node==s:
				dot.attr('node',color='blue')
				dot.node(node)
			#elif node in self.t:
			#	dot.attr('node',color='red')
			#	dot.node(node)
			elif node in word_type and node in tgt_set: #word_type[node].split('_')[1] == 'gene':
				dot.attr('node',color='red')
				dot.node(node)
			elif node in word_type and word_type[node].split('_')[1] == 'function':
				dot.attr('node',color='green')
				dot.node(node)
			elif node in word_type and word_type[node].split('_')[1] == 'symptom':
				dot.attr('node',color='yellow')
				dot.node(node)
			else:
				dot.attr('node',color='black')
				dot.node(node)
		ect = 0
		for e1,e2 in netflow_edge:
			#if e1==self.t or e2==self.t:
			#	continue
			label = ''
			#if e1 in kg_relation and e2 in kg_relation[e1]:
			#	label = 'Percha et al.'
			#else:
			#	label = 'text mining'
			if e1 not in word_type:
				t1 = 'entity'
			else:
				t1 = word_type[e1]
			if e2 not in word_type:
				t2 = 'entity'
			else:
				t2 = word_type[e2]
			dot.edge(e1,e2,label=t1+'-'+t2)
			ect += 1
		#for t in tgt_set:
		#	dot.edge(t,self.t,label='')
		#print ect,len(netflow_node)
		#for e in tgt_baseline_score:
		#	dot.edge(self.s, e, style = 'dotted' )

		dot.render(output_file)

	def plot_network_flow(self,s,word_ct,node_list, edge_list, word_type,tgt_set,output_file,net,tgt_baseline_score,multiplier):
		dot = Digraph(comment='Network flow')
		dot.format = 'eps'

		for node in node_list:
			if node in word_ct['pubmed']:
				wct = word_ct['pubmed'][node]
			else:
				wct = -1
			if node==self.s:
				dot.attr('node',color='blue')
				dot.node(node + ' '+ str(wct))
			#elif node in self.t:
			#	dot.attr('node',color='red')
			#	dot.node(node)
			elif node in word_type and node in tgt_set: #word_type[node].split('_')[1] == 'gene':
				dot.attr('node',color='red')
				dot.node(node + ' '+ str(wct))
			elif node in word_type and word_type[node].split('_')[1] == 'function':
				dot.attr('node',color='green')
				dot.node(node + ' '+ str(wct))
			elif node in word_type and word_type[node].split('_')[1] == 'disease':
				dot.attr('node',color='yellow')
				dot.node(node + ' '+ str(wct))
			else:
				dot.attr('node',color='black')
				dot.node(node + ' '+ str(wct))
		for e in edge_list:
			e1,e2,w,mw,type_list = e
			if e1==self.t or e2==self.t:
				continue
			if e1 in word_ct['pubmed']:
				wct1 = word_ct['pubmed'][e1]
			else:
				wct1 = -1
			if e2 in word_ct['pubmed']:
				wct2 = word_ct['pubmed'][e2]
			else:
				wct2 = -1
			label = ''
			#if e1 in kg_relation and e2 in kg_relation[e1]:
			#	label = 'Percha et al.'
			#else:
			#	label = 'text mining'
			if e1 not in word_type:
				t1 = 'entity'
			else:
				t1 = word_type[e1]
			if e2 not in word_type:
				t2 = 'entity'
			else:
				t2 = word_type[e2]
			edge_weight = 0
			if e1 in net and e2 in net[e1]:
				edge_weight = int(net[e1][e2])*1.0
			type = '_'.join(type_list)
			wt = '%.2f' % (float(w) / multiplier)
			mwt = '%.2f' % (float(mw) / multiplier)
			dot.edge(e1+ ' '+ str(wct1),e2+ ' '+ str(wct2),label=type+' '+t1+'-'+t2+':'+wt+'/'+mwt)
		for t in tgt_set:
			if t in word_ct['pubmed']:
				wct1 = word_ct['pubmed'][t]
			else:
				wct1 = -1
			if self.t in word_ct['pubmed']:
				wct2 = word_ct['pubmed'][self.t]
			else:
				wct2 = -1
			dot.edge(t+ ' '+ str(wct1),self.t+ ' '+ str(wct2),label='')
		#for e in tgt_baseline_score:
		#	dot.edge(self.s, e, style = 'dotted' )
		dot.render(output_file)

	def __init__(self,s,t,node_set,net,node2i,i2node,whole_network):
		self.run_network_flow(s,t,node_set,net,node2i,i2node,whole_network)
