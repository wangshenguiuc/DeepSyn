import sys
import os
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
sys.path.append(repo_dir)
os.chdir(repo_dir)

op_file = 'data/BioQA/opioid.txt'
op_set = set()
fin = open(op_file)
for line in fin:
	op_set.add(line.strip().lower())
fin.close()

from src.models.network_flow.PlotNetworkFlow import plot_network_flow
from src.models.random_walk_with_restart.RandomWalkRestart import RandomWalkRestart, DCA_vector
from src.datasets.BioNetwork import BioNetwork
import cPickle as pickle
from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.extractive_gen import ExtractGenSent
from src.datasets.SubGraph import SubGraph
from src.datasets.FindNeighbor import FindNeighbor
from src.datasets.KHopGraph import KHopGraph
import operator
import time
import collections
import numpy as np
from src.utils.evaluate.evaluate import evaluate_vec

dataset = 'drug'

min_freq_cutoff = 10

for rwr_rst in [0.8]:
	for G2G_network in ['string','literome']:
		if G2G_network == 'literome':
			literome_rwr_dump_file = 'data/network/embedding/my_dca/literome_RWR_'+str(rwr_rst)
			net_file = 'data/NLP_Dictionary/pubmed.network'
		elif G2G_network == 'string':
			literome_rwr_dump_file = 'data/network/embedding/my_dca/string_RWR_'+str(rwr_rst)
			net_file = 'data/network/human/string_integrated.txt'
		else:
			error('wrong network name')
		if os.path.isfile(literome_rwr_dump_file):
			G2G_rwr = pickle.load(open(literome_rwr_dump_file, "rb" ))
		else:
			net_file_l = []
			net_file_l.append(net_file)
			G2G_obj = BioNetwork(net_file_l,weighted=False)
			network = G2G_obj.sparse_network.toarray()
			print np.shape(network)
			#i2g = Net_obj.i2g
			#g2i = Net_obj.g2i
			#nnode = len(i2g)
			G2G_rwr = RandomWalkRestart(network, rwr_rst)
			print 'calcualte emb finished'
			with open(literome_rwr_dump_file, 'wb') as output:
				pickle.dump(G2G_rwr, output, pickle.HIGHEST_PROTOCOL)
sys.exit(-1)
literome_rwr_dump_file = 'data/network/embedding/my_dca/literome_RWR_'+str(0.8)
#net_file_l = []
#net_file_l.append(net_file)
#G2G_obj = BioNetwork(net_file_l,weighted=False)
#network = G2G_obj.sparse_network.toarray()
#G2G_obj.rwr = G2G_rwr
G2G_obj = pickle.load(open(literome_rwr_dump_file, "rb" ))
network_dump_file = 'data/Pubmed/word_network/improved_word_net_181107_' + str(min_freq_cutoff)+'_'+dataset+'_0_0'
if os.path.isfile(network_dump_file):
	ImproveNet_obj = pickle.load(open(network_dump_file, "rb" ))
else:
	Net_obj = WordNet()
	pubmed_word_net = {'data/Pubmed/word_network/predict_abst_180814':'infer','data/Pubmed/word_network/all_abst_180305':'pubmed'}
	Net_obj.ReadWordNet(pubmed_word_net,verbal=True,min_freq_cutoff=min_freq_cutoff)

	Net_obj.ReadWordType()

	stop_word_file = 'data/NLP_Dictionary/stopwords.txt'
	#stop_word_list = parse_word_net.GenerateStopWordList(Net_obj.word_ct,Net_obj.edge_ct,stop_word_file,Net_obj.word_type)
	stop_word_list = parse_word_net.GetStopWordList(stop_word_file)
	stop_word_list_manually = parse_word_net.GetStopWordList('data/NLP_Dictionary/stopwords_manually.txt')
	stop_word_list = stop_word_list.union(stop_word_list_manually)

	edge_list_l = [[dataset,'disease'],[dataset,'tissue'],[dataset,'gene'],[dataset,'function'],['tissue','function'],['tissue','disease'],['disease','function'],['disease','tissue'],['disease','disease'],['function','function'],
	['function','gene'],['tissue','gene'],['disease','gene']]
	Net_obj.ReadEdgeType(stop_word_list, edge_list_l)

	print sys.getsizeof(Net_obj)
	#selected_kg_l = [Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2f]
	#selected_kg_l = [Net_obj.Monarch_d2g,Net_obj.hpo_d2d,Net_obj.hpo_f2g,Net_obj.go_f2f,Net_obj.go_f2g]
	selected_kg_l = [Net_obj.hpo_d2d,Net_obj.go_f2f,Net_obj.go_f2g]
	ImproveNet_obj = ImprovedWordNet(Net_obj,selected_kg_l)
	Net_obj = WordNet()
	print sys.getsizeof(ImprovedWordNet)
	ImproveNet_obj.log_message('Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2g')
	ImproveNet_obj.log_message('data/Pubmed/word_network/predict_abst_180717 data/Pubmed/word_network/all_abst_180305')

	ImproveNet_obj.reload()
	with open(network_dump_file, 'wb') as output:
	   pickle.dump(ImproveNet_obj, output, pickle.HIGHEST_PROTOCOL)



gene_score_d = collections.defaultdict(dict)
gene_set = set()
file_name = ('data/network/human/InBio-Map_Symbol.sif')
fin = open(file_name)
for line in fin:
	g1,g2 = line.lower().strip().split('\t')
	gene_set.add(g1)
	gene_set.add(g2)
fin.close()

print len(gene_set)

max_path_L = 4
max_dup_edge_type = 1
max_ngh = 10


'''
GO_file_l = ['data/function_annotation/GO.network']
GO_obj = BioNetwork(GO_file_l,reverse=True)
GO_net = GO_obj.network_d[GO_file_l[0]]
GO_rev_obj = BioNetwork(GO_file_l,reverse=False)
GO_net_rev = GO_rev_obj.network_d[GO_file_l[0]]
fin = open('data/function_annotation/GO2name.txt')
GO2name  ={}
name2GO  ={}
for line in fin:
	w  = line.strip().split('\t')
	if len(w) < 2:
		continue
	GO2name[w[0]] = w[1]
	name2GO[w[1]] = w[0]
fin.close()
Func_obj = FunctionAnnotation('data/function_annotation/gene_association.goa_human', GO_net)
#nfunc = len(Func_obj.f2g)
path2gene = Func_obj.f2g
print path2gene
'''
result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/opioid/'
if not os.path.exists(result_dir):
	os.makedirs(result_dir)
else:
	for subdir, dirs, files in os.walk(result_dir):
		for file in files:
			file_path = os.path.join(subdir, file)
			os.remove(file_path)

stop_word_file = 'data/NLP_Dictionary/stopwords.txt'
#stop_word_list = parse_word_net.GenerateStopWordList(Net_obj.word_ct,Net_obj.edge_ct,stop_word_file,Net_obj.word_type)
stop_word_list = parse_word_net.GetStopWordList(stop_word_file)
stop_word_list_manually = parse_word_net.GetStopWordList('data/NLP_Dictionary/stopwords_manually.txt')
stop_word_list = stop_word_list.union(stop_word_list_manually)

#for ngh in ImproveNet_obj.net['sweating']:
#	print ngh,ImproveNet_obj.word_type[ngh],ImproveNet_obj.net['meperidine'][ngh]
#sys.exit(-1)
SenGene_obj = ExtractGenSent(working_dir = repo_dir)
op_gene_set = set(['oprm1','oprd1','oprk1','creb1','esr1','tlr4','abcb1','alb','ces1','cypc8','cyp3a4','slco1a2','ugt1a1'])
start = time.time()
topk = 10
for max_layer in [3,4]:
	for edge_wt_thres in [0.05,0.1,0.2]:
		for ci,s in enumerate(op_set):
			if s not in ImproveNet_obj.net:
				continue
			FN_obj = FindNeighbor(s,ImproveNet_obj,stop_word_list=stop_word_list)
			g2score = FN_obj.CalNgh(G2G_obj,stop_word_list=stop_word_list,max_layer=max_layer,edge_wt_thres=edge_wt_thres)

			sorted_x = sorted(g2score.items(), key=operator.itemgetter(1))
			sorted_x.reverse()
			for i in range(1000):
				g = sorted_x[i][0]
				if i>10 and g not in op_gene_set:
					continue
				print i,sorted_x[i][0],sorted_x[i][1], g in op_gene_set
				output_file = result_dir + s+'_'+str(max_layer)+'_'+str(edge_wt_thres)+'_'+g
				node_set, edge_list, node_weight = FN_obj.GetSubNetwork(g)
				plot_network_flow(output_file,s,node_set,edge_list,node_weight, ImproveNet_obj.word_ct, ImproveNet_obj.word_type)
			score = []
			label = []
			for g in g2score:
				if g in op_gene_set:
					score.append(g2score[g])
					label.append(1)
				else:
					score.append(g2score[g])
					label.append(0)
			auc,pear,spear,auprc = evaluate_vec(score,label)
			print max_layer,edge_wt_thres,s,auc,pear,spear,auprc
		#SenGene_obj.get_graph_sentence(edge_list,image_file,Net_obj)
'''
wct = {}
file_l = ['data/Pubmed/word_network/predict_abst_180814','data/Pubmed/word_network/all_abst_180305']
for file in file_l:
	fin = open(file)
	for line in fin:
		w = line.lower().split('\t')
		w1 = w[0]
		w2 = w[2]
		if w1 in op_set or w2 in op_set:
			wct[w1] = wct.get(w1,0) + 1
			wct[w2] = wct.get(w2,0) + 1
	fin.close()
ww = sorted(wct.items(), key = lambda x: x[1], reverse=True)
for i in range(50):
	if ww[i][0] not in op_set:
		print ww[i][0],ww[i][1]
'''
