import numpy as np
import collections
import operator
import sys
class ImprovedWordNet():



	def ValidPath(self,s,t):
		if len(self.net[s])==0 or  len(self.bp_net[t])==0:
			return False
		return True

	def add_user_define_edge(self,s,t):
		self.net[s][t]['manually'] = 100000
		self.bp_net[s][t]['manually'] = 100000

	def log_message(self,s):
		self.message += s + '\n'


	def valid_edge_type(self, k1, k2):
		if k2 not in self.word_type or k1 not in self.word_type:
			return False
		if [self.word_type[k1].split('_')[1],self.word_type[k2].split('_')[1]] not in self.graph_edge_type:
			return False
		return True

	def GenerateImprovedNetwork(self,kg_list,WN,odds_cutoff=10,select_edge_type=True,max_ngh=100):

		for source in WN.edge_ct:
			max_value = -1
			for ct,k1 in enumerate(WN.edge_ct[source]):
				if k1 in WN.stop_word_list:
					continue
				new_d = {}
				for k2 in WN.edge_ct[source][k1]:
					if k2 in WN.stop_word_list:
						continue
					if select_edge_type and not self.valid_edge_type(k1,k2):
						continue
					value = WN.edge_ct[source][k1][k2]
					#value = WN.edge_ct[source][k1][k2]*1.0 / WN.word_ct[source][k2] / WN.word_ct[source][k1]*WN.total_wt[source]
					if value>max_value:
						print source,value,WN.total_wt[source],k1,k2,WN.word_ct[source][k2],WN.edge_ct[source][k1][k2],WN.word_ct[source][k1]
						max_value = value
					if value <= 0:
						continue
					#table = np.zeros((2,2))
					#table[0,1] = wct[k2] - ect[k1][k2]
					#table[1,0] = wct[k1] - ect[k1][k2]
					#table[1,1] = ect[k1][k2]
					#table[0,0] = sent_ct - table[0,1] - table[1,1] - table[1,0]
					#oddsratio, pvalue = stats.fisher_exact([[8, 2], [1, 5]])
					#new_d[k2] = - np.log10(pvalue)
					#if ect[k1][k2]*1.0 / wct[k2]<0.01:
					#    continue
					new_d[k2] = value
				sort_x = sorted(new_d.items(),key=operator.itemgetter(1))
				sort_x.reverse()
				wt_sum = {}
				for i in range(min(len(sort_x),max_ngh)):
					k2,sc = sort_x[i]
					wt = self.word_type[k2].split('_')[1]
					wt_sum[wt] = wt_sum.get(wt,0) + sc
				for i in range(min(len(sort_x),max_ngh)):
					k2,sc = sort_x[i]
					wt = self.word_type[k2].split('_')[1]
					if k1 not in self.net:
						self.net[k1] = collections.defaultdict(dict)
					if k2 not in self.bp_net:
						self.bp_net[k2] = collections.defaultdict(dict)
					self.net[k1][k2][source] = sc / wt_sum[wt]
					self.bp_net[k2][k1][source] =sc / wt_sum[wt]
				if ct%100000==0:
					print 'generate graph finished',ct*1.0/len(WN.edge_ct[source])
					sys.stdout.flush()
		for dr in kg_list:
			for g in dr:
				for t in dr[g]:
					if select_edge_type and not self.valid_edge_type(g,t):
						continue
					if g not in self.net:
						self.net[g] = collections.defaultdict(dict)
					if t not in self.bp_net:
						self.bp_net[t] = collections.defaultdict(dict)
					self.net[g][t]['database'] = 1.#dr[g][t]
					self.bp_net[t][g]['database'] = 1.#dr[g][t]


		return self.net,self.bp_net

	def reload(self):
		self.dis2source = collections.defaultdict(dict)
		self.dis2target = collections.defaultdict(dict)

	def __init__(self,WN,kg_list,max_ngh=20):
		self.net = {}
		self.bp_net = {}
		self.graph_edge_type = WN.graph_edge_type
		print self.graph_edge_type
		self.candidate_gene = WN.candidate_gene
		self.word_ct = WN.word_ct
		self.word_type = WN.word_type
		self.sent_ct = 239833652
		self.message = ''
		self.GenerateImprovedNetwork(kg_list,WN,max_ngh=max_ngh)
		self.dis2source = collections.defaultdict(dict)
		self.dis2target = collections.defaultdict(dict)
