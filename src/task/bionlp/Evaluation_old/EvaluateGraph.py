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

for use_gene_2_gene in [0,1]:
    for use_direct_edge in [0,1]:
        network_dump_file = 'data/Pubmed/word_network/improved_word_net_180814_' + str(min_freq_cutoff)+'_'+dataset+'_'+str(use_direct_edge)+'_'+str(use_gene_2_gene)
        print network_dump_file
        if os.path.isfile(network_dump_file):
            ImproveNet_obj = pickle.load(open(network_dump_file, "rb" ))
            print network_dump_file
        else:
            sys.exit('not exist file')
            Net_obj = WordNet()
            pubmed_word_net = {'data/Pubmed/word_network/predict_abst_180814':'infer','data/Pubmed/word_network/all_abst_180305':'pubmed'}
            Net_obj.ReadWordNet(pubmed_word_net,verbal=True,min_freq_cutoff=min_freq_cutoff)

            Net_obj.ReadWordType()
            edge_list_l = [[dataset,'symptom'],[dataset,'tissue'],[dataset,'disease'],[dataset,'function'],['tissue','function'],['tissue','symptom'],['symptom','function'],['symptom','tissue'],
                ['function','function'],['function','gene']]
            if use_direct_edge:
                edge_list_l.append([dataset,'gene'])
            if use_gene_2_gene:
                edge_list_l.append(['gene','gene'])
            Net_obj.ReadEdgeType(stop_word_list, edge_list_l)

            selected_kg_l = [Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2f,Net_obj.go_f2g]

            ImproveNet_obj = ImprovedWordNet(Net_obj,selected_kg_l)
            print sys.getsizeof(ImprovedWordNet)
            ImproveNet_obj.log_message('Net_obj.literome_g2g,Net_obj.synonym_g2g,Net_obj.hpo_d2d,Net_obj.go_f2')
            ImproveNet_obj.log_message('data/Pubmed/word_network/predict_abst_180717 data/Pubmed/word_network/all_abst_180305')
            ImproveNet_obj.reload()
            with open(network_dump_file, 'wb') as output:
                pickle.dump(ImproveNet_obj, output, pickle.HIGHEST_PROTOCOL)


        process = psutil.Process(os.getpid())
        print process.memory_info().rss

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
        for max_path_L in [5,6,4,7]:
            for max_dup_edge_type in [2,3]:
                for max_ngh in [10,20,50]:
                    result_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/result/network_flow/'+dataset_name+'/'+str(max_path_L)+'_'+str(max_dup_edge_type)+'_'+str(max_ngh)+'_'+str(min_freq_cutoff)+'_'+str(use_direct_edge)+'_'+str(use_gene_2_gene)+'/'
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                    G2KhopGraph = {}
                    D2KhopGraph = {}
                    auc_our = 0.5
                    auc_baseline = 0.5
                    start = time.time()
                    for ci,ss in enumerate(monarch_d2g.keys()):
                        s = ss.replace('_',' ').replace('  ',' ')
                        if dataset_name == 'MonarchDisease':
                            s = ''.join([i for i in s if not i.isdigit()]).strip()
                        if ci%total_pid != pid:
                            continue
                        #print s,ss
                        #if s not in D2KhopGraph:
                        D2KhopGraph = KHopGraph(ImproveNet_obj.net)
                        D2KhopGraph.get_K_ngh(s)
                        if len(D2KhopGraph.dis2source[s])==1:
                            continue
                        print 'calculate',s
                        file = result_dir+s+'.txt'
                        print file
                        if os.path.isfile(file):
                            tmp_line = ''
                            fin = open(file)
                            for line in fin:
                                tmp_line = line
                            fin.close()
                            #if tmp_line.startswith('FINAL_AUC'):
                            #    continue
                            #continue
                        fout = open(file,'w')
                        process = psutil.Process(os.getpid())
                        print 'memory',process.memory_info().rss
                        sys.stdout.flush()
                        #print s,len(monarch_d2g[s]),len(D2KhopGraph.dis2source[s])
                        predict = []
                        baseline = []
                        truth = []
                        pos_gene = set(monarch_d2g[ss].keys())
                        neg_gene = candidate_gene - pos_gene
                        pos_gene = list(pos_gene)
                        #print pos_gene
                        pos_gene.extend(list(neg_gene))
                        #print pos_gene[:5]

                        G2KhopGraph = KHopGraph(ImproveNet_obj.bp_net)
                        npos_gene = 0
                        for i,t in enumerate(pos_gene):
                            #if t not in G2KhopGraph:

                            if t not in monarch_d2g[ss] and npos_gene==0:
                                continue
                            G2KhopGraph.get_K_ngh(t)
                            if len(G2KhopGraph.dis2source[t])==1:
                                continue
                            sub_graph = SubGraph(ImproveNet_obj,D2KhopGraph,G2KhopGraph,s,set([t]),max_ngh=max_ngh)
                            if len(sub_graph.graph_node_set) <= 1:
                                continue
                            if t not in monarch_d2g[ss]:
                                a = np.random.rand(1)
                                if a > 0.3:
                                    continue
                            gene_score[s][t],confidence = sub_graph.CalSubGraphScore(ImproveNet_obj,max_dup_edge_type = max_dup_edge_type,dfs_max_depth=max_path_L)[:2]
                            #if gene_score[s][t] > -1 and npath[s][t]>1:
                                #print s,t,gene_score[s][t],npath[s][t],t in monarch_d2g[ss],max_ngh
                            #    pass
                                #SenGene_obj.get_graph_sentence(edge_list,image_file,Net_obj)
                            #print i,len(candidate_gene),t in monarch_d2g[ss]
                            predict.append(gene_score[s][t])
                            if gene_score[s][t] > -1:
                                if t in monarch_d2g[ss].keys():
                                    npos_gene+=1
                            if s in ImproveNet_obj_baseline.net and t in ImproveNet_obj_baseline.net[s]:
                                baseline_score = ImproveNet_obj_baseline.net[s][t]['pubmed']
                                #print s,t,baseline_score
                            else:
                                baseline_score = -1

                            baseline.append(baseline_score)
                            truth.append(int(t in monarch_d2g[ss]))
                            auc_our = evaluate_vec(predict,truth)[0]
                            auc_baseline = evaluate_vec(baseline,truth)[0]
                            end = time.time()
                            fout.write(str(i*1.0/len(pos_gene))+'\t'+str(ss)+'\t'+str(t)+'\t'+str(t in monarch_d2g[ss])+'\t'+str(gene_score[s][t])+'\t'+str(baseline_score)+'\t'+str(auc_our)+'\t'+str(auc_baseline)+'\t'+str(end - start)+'\n')
                            fout.flush()
                            print i*1.0/len(pos_gene), npos_gene,ss, t, t in monarch_d2g[ss], gene_score[s][t],baseline_score,auc_our,auc_baseline,end - start
                            sys.stdout.flush()
                        print "FINAL_AUC:",auc_our,auc_baseline,s
                        fout.write("FINAL_AUC: "+str(auc_our)+' '+str(auc_baseline)+' '+str(s)+'\n')
                        fout.flush()
                        fout.close()
