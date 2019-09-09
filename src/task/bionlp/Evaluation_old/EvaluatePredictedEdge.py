
import sys
import os
repo_dir = '/srv/local/work/swang141/Sheng_repo/'
nlp_tool_dir = '/srv/local/work/swang141/PatientSetAnnotation/Sheng/analysis/stanford_parser_java/stanford-parser-full-2017-06-09/'
pubmed_dir = '/srv/local/work/swang141/PatientSetAnnotation/Sheng/data/pubmed/'
sys.path.append(repo_dir)
os.chdir(repo_dir)

from src.datasets.WordNet import WordNet
from src.datasets.ImprovedWordNet import ImprovedWordNet
from src.utils.nlp import parse_word_net
from src.utils.evaluate.evaluate import evaluate_vec
from src.models.generate_sentence.extractive_gen import ExtractGenSent
from src.datasets.SubGraph import SubGraph
import numpy as np
import operator
import collections

pubmed_word_net = 'data/Pubmed/word_network/predict_abst_180717'
#pubmed_word_net = 'data/Pubmed/word_network/all_abst_180305'
Net_obj = WordNet()

Net_obj.ReadWordNet(pubmed_word_net,verbal=True,min_freq_cutoff=0)
Net_obj.ReadWordType()
stop_word_file = 'data/NLP_Dictionary/stopwords.txt'
#stop_word_list = parse_word_net.GenerateStopWordList(Net_obj.word_ct,Net_obj.edge_ct,stop_word_file,Net_obj.word_type)
stop_word_list = parse_word_net.GetStopWordList(stop_word_file)

stop_word_list_manually = parse_word_net.GetStopWordList('data/NLP_Dictionary/stopwords_manually.txt')
stop_word_list = stop_word_list.union(stop_word_list_manually)
Net_obj.ReadWordType()
Net_obj.ReadEdgeType(stop_word_list)

go2gene_set = {}
all_genes = set()
fin = open('data/NLP_Dictionary/go2gene_set_complete.txt')
for line in fin:
    w = line.strip().split('\t')
    go = w[0]
    if go not in Net_obj.edge_ct:
        continue
    go2gene_set[go] = set()
    for i in range(1, len(w)):
        go2gene_set[go].add(w[i])
        all_genes.add(w[i])
fin.close()

go2gene_sc = collections.defaultdict(dict)
for go in go2gene_set:
    for g in all_genes:
        go2gene_sc[go][g] = Net_obj.edge_ct[go].get(g,0)

auc_mean = []
for go in go2gene_set:
    pred = []
    truth = []
    for g in all_genes:
        pred.append(go2gene_sc[go][g])
        if g in go2gene_set[go]:
            truth.append(1)
        else:
            truth.append(0)
    auc = evaluate_vec(pred,truth)[0]
    if auc!=0.5:
        print go, auc, np.mean(auc_mean)
        auc_mean.append(auc)
