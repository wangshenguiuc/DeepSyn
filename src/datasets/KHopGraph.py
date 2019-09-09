import numpy as np
import collections
import copy
import operator
from src.models.network_flow.NetworkFlow import NetworkFlow


class KHopGraph():

	def __init__(self, net,stop_word_list=[],max_path_L = 4,edge_wt_thres = 1,max_ngh=10,max_inf=1000000):
		self.dis2source =  collections.defaultdict(dict)
		self.prev = {}
		self.max_path_L = max_path_L
		self.net = net
		self.stop_word_list=stop_word_list
		#self.get_K_ngh(s,net,max_path_L,edge_wt_thres)


	def get_K_ngh(self,s):
		unvisited = set()
		unvisited.add(s)
		self.dis2source[s][s] = 0
		visited = set()
		while len(unvisited)!=0:
			unvisited_loop = set(unvisited)
			for ss in unvisited_loop:
				if ss in self.stop_word_list:
					continue
				visited.add(ss)
				if self.dis2source[s][ss] > self.max_path_L:
					continue
				if ss not in self.net:
					continue
				for ngh in self.net[ss]:
					if ngh not in visited and ngh not in unvisited:
						unvisited.add(ngh)
						self.dis2source[s][ngh] = self.dis2source[s][ss] + 1
						self.prev[ngh] = ss
			unvisited = unvisited - visited


