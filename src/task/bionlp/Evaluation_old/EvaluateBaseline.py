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
import psutil
from src.utils.evaluate.evaluate import evaluate_vec


pid = int(sys.argv[1])
dataset = str(sys.argv[2])
total_pid = int(sys.argv[3])

min_freq_cutoff = 12


stop_word_file = 'data/NLP_Dictionary/stopwords.txt'
#stop_word_list = parse_word_net.GenerateStopWordList(Net_obj.word_ct,Net_obj.edge_ct,stop_word_file,Net_obj.word_type)
stop_word_list = parse_word_net.GetStopWordList(stop_word_file)
stop_word_list_manually = parse_word_net.GetStopWordList('data/NLP_Dictionary/stopwords_manually.txt')
stop_word_list = stop_word_list.union(stop_word_list_manually)

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

process = psutil.Process(os.getpid())
print process.memory_info().rss
sys.stdout.flush()
gene_score = collections.defaultdict(dict)
npath = collections.defaultdict(dict)

Net_obj_baseline = WordNet()
if dataset == 'disease':
    dataset_name = 'MonarchDisease'
    candidate_gene,monarch_d2g = Net_obj_baseline.ReadNodeTypeMonarchDiseaseGene()
    print len(monarch_d2g)
elif dataset == 'drug':
    dataset_name = 'CTRPdrugGene'
    candidate_gene,monarch_d2g = Net_obj_baseline.ReadNodeTypeCtrpDrugGene()
    print len(monarch_d2g)
else:
    sys.exit('wrong dataset')
sys.stdout.flush()


sys.stdout.flush()
'''
ct = 0
for s in ImproveNet_obj_baseline.net:
    for t in ImproveNet_obj_baseline.net[s]:
        print s,t,ImproveNet_obj_baseline.net[s][t]
        if ct>100:
            sys.exit('end')
        ct += 1
'''

result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'+str(min_freq_cutoff)+'/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
fout = open(result_dir+dataset+'.txt','w')
auc_our = 0.5
auc_baseline = 0.5
start = time.time()
for ci,ss in enumerate(monarch_d2g.keys()):
    s = ss.replace('_',' ').replace('  ',' ')
    if dataset_name == 'MonarchDisease':
        s = ''.join([i for i in s if not i.isdigit()]).strip()
    #if ci%total_pid != pid:
    #    continue
    #print s,ss
    #if s not in D2KhopGraph:
    #print 'calculate',s
    process = psutil.Process(os.getpid())
    #print 'memory',process.memory_info().rss
    sys.stdout.flush()
    #print s,len(monarch_d2g[s]),len(D2KhopGraph.dis2source[s])
    predict = []
    baseline = []
    truth = []
    pos_gene = set(monarch_d2g[ss].keys())
    neg_gene = candidate_gene - pos_gene
    pos_gene = list(pos_gene)
    pos_gene.extend(list(neg_gene))

    for i,t in enumerate(pos_gene):
        if s in ImproveNet_obj_baseline.net and t in ImproveNet_obj_baseline.net[s]:
            baseline_score = ImproveNet_obj_baseline.net[s][t]['pubmed']
        else:
            baseline_score = -1
        baseline.append(baseline_score)
        #
        if baseline_score > -1:
            print s,t,baseline_score
            fout.write(s+'\t'+t+'\t'+str(baseline_score)+'\n')
    #print "FINAL_AUC:",auc_our,auc_baseline,s
fout.close()
