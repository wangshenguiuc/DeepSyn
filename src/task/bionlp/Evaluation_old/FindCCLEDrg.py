import sys
import os
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'


sys.path.append(repo_dir)
os.chdir(repo_dir)

import cPickle as pickle
from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.models.generate_sentence.extractive_gen import ExtractGenSent
from src.datasets.SubGraph import SubGraph
from src.datasets.KHopGraph import KHopGraph
import operator
import time
import collections
import numpy as np
from src.utils.evaluate.evaluate import evaluate_vec
#dataset = sys.argv[1]
dataset = 'drug'
min_freq_cutoff = 12
network_dump_file = 'data/Pubmed/word_network/improved_word_net_181009_' + str(min_freq_cutoff)+'_'+dataset+'_0_0'

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

    edge_list_l = [[dataset,'symptom'],[dataset,'tissue'],[dataset,'disease'],[dataset,'function'],['tissue','function'],['tissue','symptom'],['symptom','function'],['symptom','tissue'],
        ['function','function'],['function','gene'],['gene','gene']]
    Net_obj.ReadEdgeType(stop_word_list, edge_list_l)

    print sys.getsizeof(Net_obj)
    selected_kg_l = [Net_obj.literome_g2g,Net_obj.PPI_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2f]
    ImproveNet_obj = ImprovedWordNet(Net_obj,selected_kg_l)
    Net_obj = WordNet()
    print sys.getsizeof(ImprovedWordNet)
    ImproveNet_obj.log_message('Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2')
    ImproveNet_obj.log_message('data/Pubmed/word_network/predict_abst_180717 data/Pubmed/word_network/all_abst_180305')

    ImproveNet_obj.reload()
    with open(network_dump_file, 'wb') as output:
       pickle.dump(ImproveNet_obj, output, pickle.HIGHEST_PROTOCOL)


baseline_network_dump_file = 'data/Pubmed/word_network/baseline_improved_word_net_180814_' + str(min_freq_cutoff)+'_'+dataset
if os.path.isfile(baseline_network_dump_file):
	ImproveNet_obj_baseline = pickle.load(open(baseline_network_dump_file, "rb" ))
else:
	sys.exit('not exist file')
	edge_list_l = [[dataset,'gene']]
	Net_obj_baseline = WordNet()
	pubmed_word_net = {'data/Pubmed/word_network/all_abst_180305':'pubmed'}
	Net_obj_baseline.ReadWordNet(pubmed_word_net,verbal=True,min_freq_cutoff=min_freq_cutoff)

	Net_obj_baseline.ReadWordType()
	Net_obj_baseline.ReadEdgeType(stop_word_list, edge_list_l)
	ImproveNet_obj_baseline = ImprovedWordNet(Net_obj_baseline,[])
	print sys.getsizeof(ImproveNet_obj_baseline)
	ImproveNet_obj_baseline.reload()
	with open(baseline_network_dump_file, 'wb') as output:
		pickle.dump(ImproveNet_obj_baseline, output, pickle.HIGHEST_PROTOCOL)


import psutil
sys.stdout.flush()

dataset_name = 'ctrp_correlated_genes'

fin = open('data/drug/ctrp/drug_map.txt')
d2dname = {}
for line in fin:
    w = line.strip().split('\t')
    d2dname[w[2]] = w[0]
fin.close()

fin = open('data/NLP_Dictionary/top_genes_exp_hgnc.txt')
d2g = {}
for line in fin:
    w = line.strip().split('\t')
    cor = float(w[2])
    if abs(cor) < 0.5:
        continue
    d = d2dname[w[1]]
    if d not in d2g:
        d2g[d] = set()
    d2g[d].add(w[0].lower())
fin.close()

max_path_L = 5
max_dup_edge_type = 2
max_ngh = 10

result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'+str(max_path_L)+'_'+str(max_dup_edge_type)+'_'+str(max_ngh)+'graph_figure/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
else:
    for subdir, dirs, files in os.walk(result_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            os.remove(file_path)


net = ImproveNet_obj.net

start = time.time()
for ci,ss in enumerate(d2g.keys()):
    s = ss.lower().split(' ')[0]
    print s
    D2KhopGraph = KHopGraph(ImproveNet_obj.net)
    D2KhopGraph.get_K_ngh(s)
    #print len(D2KhopGraph.dis2source[s])
    if len(D2KhopGraph.dis2source[s])==1:
        continue
    tgt_set = set()
    G2KhopGraph = KHopGraph(ImproveNet_obj.bp_net)
    for i,t in enumerate(d2g[s]):
        #if t not in G2KhopGraph:
        G2KhopGraph.get_K_ngh(t)
        if len(G2KhopGraph.dis2source[t])==1:
            continue
        tgt_set.add(t)

    #continue
    image_file = [result_dir,s]
    sub_graph = SubGraph(ImproveNet_obj,D2KhopGraph,G2KhopGraph,s,set(tgt_set),max_ngh=max_ngh)
    if len(sub_graph.graph_node_set) <= 1:
        continue

    tgt_baseline_score = {}
    for t in tgt_set:
        if s in ImproveNet_obj_baseline.net and t in ImproveNet_obj_baseline.net[s]:
            baseline_score = ImproveNet_obj_baseline.net[s][t]['pubmed']
            #print s,t,baseline_score
            tgt_baseline_score[t] = baseline_score
        else:
            baseline_score = -1


    gene_score,confidence,confidence_detail,npath,edge_list,node_list,nfind_tgt = sub_graph.CalSubGraphScore(ImproveNet_obj,image_file = image_file,
    max_dup_edge_type = max_dup_edge_type,dfs_max_depth=max_path_L,tgt_baseline_score=tgt_baseline_score)
    print s, len(d2g[s]), len(tgt_set), nfind_tgt, len(node_list),node_list
    if nfind_tgt>5:
        sys.exit()
    if gene_score>-1 and confidence>0:
        #print s,sub_graph.word_type[s],tgt_set,gene_score,confidence,confidence_detail
        sys.stdout.flush()
        #SenGene_obj.get_graph_sentence(edge_list,image_file,Net_obj)
