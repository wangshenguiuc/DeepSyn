import sys
import os
import numpy as np
import networkx as nx
import pickle
from utils.configure import *
from utils.utils import read_ImprovedNet_obj, read_network_data
#from utils.PlotNetworkFlow import plot_network_flow
from utils.FindLayer import FindLayer

def load_data(DATA_DIR, pubmed_file, deepsyn_file, net_file, network_rwr):
	G2G_obj,network_gene_list = read_network_data(net_file_l = net_file,network_rwr = network_rwr)
	KnowledgeGraph_obj, stop_word_list = read_ImprovedNet_obj(pubmed_file = pubmed_file, deepsyn_file=deepsyn_file, DATA_DIR=DATA_DIR)

	return network_gene_list,G2G_obj,KnowledgeGraph_obj,stop_word_list

def query_graph(st_terms, ed_terms, kw_st, kw_ed, Graph_obj, network_gene_list,G2G_obj,stop_word_list,max_layer=3,edge_wt_thres=0.01,net_topk=2,prop=True,exclude_edge_type=[],max_end_nodes=100):
	''''
	ss: disease name or drug name
	gene_set: a set of genes
	G2G_obj: Gene network object
	network_gene_list: list of genes in the network
	KnowledgeGraph_obj: knowledge graph
	stop_word_list: stop words for NLP

	return:
	node_set: a set of nodes in the subgraph
	edge_list: a list of edges in the subgraph
	node_weight: weight of each node
	'''
	#print 'here'

	FN_obj = FindLayer(st_terms,ed_terms,Graph_obj,kw_st, kw_ed,stop_word_list=stop_word_list,net_topk=net_topk,max_layer = max_layer,edge_wt_thres =edge_wt_thres,max_end_nodes=max_end_nodes,prop=prop,exclude_edge_type=exclude_edge_type,DATA_DIR=DATA_DIR)
	node_set,edge_list,node_weight = FN_obj.CalNgh(G2G_obj,all_type_same_weight=True,use_direct_gene_edge=True)
	return node_set,edge_list,node_weight,FN_obj

def expand_term(word, all_term, word_ct, min_word_ct = 1000):
	res_term = set()
	for w in all_term:
		if word in w and word_ct[w] > min_word_ct:
			res_term.add(w)
	return res_term

def parse_query(field2term, KnowledgeGraph_obj, network_gene_list,kw_list = ['disease', 'function', 'gene', 'drug']):
	expanded_query_set = set()
	for i in range(len(kw_list)):
		if field2term[kw_list[i]]!='*':
			continue
		for j in range(i+1,len(kw_list)):
			if field2term[kw_list[j]]!='*' and field2term[kw_list[j]]!='':
				field2term[kw_list[i]] = ''

	all_term = {}
	word_ct = {}
	for w in KnowledgeGraph_obj.word_type:
		tp = KnowledgeGraph_obj.word_type[w].split('_')[1]
		if tp not in all_term:
			all_term[tp] = set()
		all_term[tp].add(w)
		sum = 0
		for s in KnowledgeGraph_obj.word_ct:
			sum += KnowledgeGraph_obj.word_ct[s].get(w,0)
		word_ct[w] = sum
	all_term['gene'] = set(network_gene_list)
	query = {}
	for field in field2term:
		term = field2term[field]
		if term == '*':
			query[field] = all_term[field]
		elif term == '':
			continue
		else:
			query[field] = set()
			for w in term:
				if w not in all_term[field]:
					expand_w = expand_term(w, all_term[field], word_ct)
					for ww in expand_w:
						query[field].add(ww)
				else:
					query[field].add(w)
			for qq in query[field]:
				expanded_query_set.add(qq)
		#print field, len(query[field])
	return query, expanded_query_set


def run_query(query, nmax_node, network_gene_list,G2G_obj,KnowledgeGraph_obj,stop_word_list,write_graph2file=False,net_topk_l = [4,3,2,1],edge_wt_thres_l = [0.01], max_layer_l = [4,3,2], DATA_DIR = ''):
	kw_list = ['disease', 'function', 'gene', 'drug']
	max_nnode = {}
	for kw in kw_list:
		max_nnode[kw] = max(nmax_node,len(query[kw]))
	query_terms, expanded_query_set = parse_query(query, KnowledgeGraph_obj, network_gene_list, kw_list)
	whole_graph = []
	return_graph_obj = []
	whole_node_set = set()
	whole_node_weight = {}
	whole_node_pvalue = {}
	#layer to layer max flow
	cur_ed = 0
	for i,kw_st in enumerate(kw_list):
		if kw_st not in query_terms or len(query_terms[kw_st])==0:
			continue
		if i < cur_ed:
			continue
		for j,kw_ed in enumerate(kw_list):
			if kw_ed not in query_terms or len(query_terms[kw_ed])==0:
				continue
			if j<=i:
				continue
			cur_ed = j
			max_layer = max_layer_l[0]
			edge_wt_thres = edge_wt_thres_l[0]
			net_topk = net_topk_l[0]
			#find the end nodes of this layer
			node_set,edge_list,node_weight,_ = query_graph(query_terms[kw_st], query_terms[kw_ed], kw_st, kw_ed, KnowledgeGraph_obj, network_gene_list, G2G_obj, stop_word_list,
			max_layer=max_layer,edge_wt_thres=edge_wt_thres,net_topk=net_topk,prop=False,exclude_edge_type=[['disease','disease'],['disease','gene']])

			g2sc = {}
			for edge in edge_list:
				e1,e2,sc,tp = edge
				if e2 not in KnowledgeGraph_obj.word_type:
					continue
				tp2 = KnowledgeGraph_obj.word_type[e2].split('_')[1]
				if tp2 == kw_ed:
					g2sc[e2] = sc
			if len(query_terms[kw_ed]) <= max_nnode[kw_ed]:
				for g in query_terms[kw_ed]:
					g2sc[g] = 1000000
			select_node_list = []
			g2sc_list = [(k, g2sc[k]) for k in sorted(g2sc, key=g2sc.get, reverse=True)]

			for i in range(min(len(g2sc_list),max_nnode[kw_ed])):
				g,v = g2sc_list[i]
				select_node_list.append(g)
			print (select_node_list)
			#iterate over hyperparameters to find the best graph in this layer
			find_valid = False
			best_graph = []
			best_ct = 0.
			best_layer = 0
			for net_topk in net_topk_l:
				for edge_wt_thres in edge_wt_thres_l:
					for max_layer in max_layer_l:
						node_set,edge_list,node_weight,FN_obj = query_graph(query_terms[kw_st], select_node_list , kw_st, kw_ed, KnowledgeGraph_obj, network_gene_list, G2G_obj, stop_word_list,
						max_layer=max_layer,edge_wt_thres=edge_wt_thres,net_topk=net_topk,prop=False,exclude_edge_type=[['disease','disease'],['disease','gene']])

						tp2ct = {}
						for n in node_set:
							if n not in KnowledgeGraph_obj.word_type:
								continue
							tp = KnowledgeGraph_obj.word_type[n].split('_')[1]
							tp2ct[tp] = tp2ct.get(tp, 0) +1

						valid = True
						for tp in tp2ct:
							if tp2ct[tp] > max_nnode[tp]:
								valid = False
						if valid and np.sum(list(tp2ct.values())) > best_ct or ( np.sum(list(tp2ct.values())) == best_ct and max_layer > best_layer):
							#print 'valid',net_topk, edge_wt_thres, max_layer, tp2ct, best_layer, best_ct
							best_ct = np.sum(list(tp2ct.values()))
							best_graph = [node_set,edge_list,node_weight]

							best_layer = max_layer
							best_FN_obj = FN_obj

			#add the best graph in this layer to the whole graph
			node_set,edge_list,node_weight = best_graph

			#node_pvalue = best_FN_obj.CalPvalue(node_set, G2G_obj)
			node_pvalue = node_weight
			query_terms[kw_ed] = set()
			for w in node_set:
				tp = KnowledgeGraph_obj.word_type[w].split('_')[1]
				if tp==kw_ed:
					query_terms[kw_ed].add(w)

			for w in node_set:
				whole_node_set.add(w)
				whole_node_weight[w] = node_weight[w]
				whole_node_pvalue[w] = node_pvalue[w]

			whole_graph.extend(edge_list)
			#graph_sent_list = SenGene_obj.get_graph_sentence(edge_list,'')
			return_graph_obj.extend(edge_list)
			break
	if write_graph2file:
		output_file = '/cellar/users/majianzhu/Data/wangsheng/NetAnt/src/task/bionlp/server/'+'inflam_' + str(max_nnode['function'])
		plot_network_flow(output_file,[],whole_node_set,whole_graph,whole_node_weight, KnowledgeGraph_obj.word_ct, KnowledgeGraph_obj.word_type)

	#generate networkx object
	G = nx.DiGraph()
	for w in whole_node_weight:
		G.add_node(w, pvalue=node_pvalue[w])
		G.add_node(w,confidence=whole_node_weight[w])
		G.add_node(w, type=KnowledgeGraph_obj.word_type[w])
		if w in expanded_query_set:
			G.add_node(w, inquery=True)
		else:
			G.add_node(w, inquery=False)
	for e in return_graph_obj:
		#e1,e2,type_list,pm_title,pm_id,sent = e
		#G.add_edge(e1, e2, type=type_list, pm_title=pm_title, pm_id=pm_id, sent=sent)
		e1,e2,wt,type_list = e
		G.add_edge(e1, e2, type=type_list)

	#print nmax_node,G.nodes(data=True)
	#
	return G

def write_to_cyto_scape(G_obj,output_file):
	pvalue = nx.get_edge_attributes(G_obj,'type')
	fout = open(output_file+'.edge','w')
	fout.write('source\ttarget\n')
	for edges in G_obj.edges:
		n1,n2 = edges
		fout.write(n1+'\t'+n2+'\t'+pvalue[(n1,n2)]+'\n')
	fout.close()

	pvalue = nx.get_node_attributes(G_obj,'pvalue')
	type = nx.get_node_attributes(G_obj,'type')
	confidence = nx.get_node_attributes(G_obj,'confidence')
	fout = open(output_file+'.node','w')
	fout.write('node\ttype\ttype\tgene_score\n')
	for n in G_obj.nodes:
		fout.write(n+'\t'+type[n]+'\t'+str(pvalue[n])+'\t'+str(confidence[n])+'\n')
	fout.close()

if len(sys.argv) <= 2:
	query = {}
	query['disease'] = ['fanconi anemia']
	#query['disease'] = ['inflamm']
	#query['gene'] = '*'#['ghrl', 'sele', 'cyp19a1', 'ppard', 'vimp', 'gstp1', 'osmr', 'fabp4', 'il1b', 'chrfam7a']#'*'
	query['gene'] =['recql5', 'fancd2', 'blm', 'wrn', 'fancc', 'fanci', 'fancm', 'sptan1', 'bub1', 'klhdc8b', 'miip', 'fancf', 'fancg', 'slx4', 'fancl', 'fance', 'fan1', 'rad51', 'xpa', 'brca1']
	query['drug'] = 'Aspirin'
	query['function'] = ['inflammasome', 'Golgi apparatus', 'Extracellular space', 'Mitochondria']
	nmax_node = 30
else:
	query = {}
	query['drug'] = sys.argv[1].lower().split('#')
	query['disease'] = sys.argv[2].lower().split('#')
	query['gene'] = sys.argv[3].lower().split('#')
	query['function'] = sys.argv[4].lower().split('#')
	nmax_node= int(sys.argv[5])

network_gene_list,G2G_obj,KnowledgeGraph_obj,stop_word_list =load_data(DATA_DIR = DATA_DIR, deepsyn_file= DATA_DIR + 'network/all_abst_181110',pubmed_file= DATA_DIR + 'network/predict_abst_181110', net_file= DATA_DIR + 'network/string_integrated.txt', network_rwr=DATA_DIR + 'network/string_RWR_0.8')

G_obj = run_query(query, nmax_node, network_gene_list,G2G_obj,KnowledgeGraph_obj,stop_word_list,DATA_DIR=DATA_DIR)
print (G_obj.edges(data=True))
print (G_obj.nodes(data=True))
write_to_cyto_scape(G_obj,query['disease'][0]+str(nmax_node))
#
