import sys
import os
import networkx as nx
import argparse
from utils import *
import warnings


parser = argparse.ArgumentParser(description='parameters for DeepSyn')
parser.add_argument('--data_dir', default='DeepSyn/data/knowledge_network/', help='path of the data folder downloaded from dropbox')
parser.add_argument('--disease',  help='diseases splited by ";". e.g., breast cancer;lung cancer', default='')
parser.add_argument('--drug',  help='diseases splited by ";". e.g., doxurbicin;etoposide', default='')
parser.add_argument('--gene', help='diseases splited by ";". e.g., tp53;kmt2a', default='')
parser.add_argument('--function',  help='diseases splited by ";". e.g., dna repair', default='')
parser.add_argument('--output_file',  help='output file of edges and nodes', default='output.txt')
args = parser.parse_args()

DATA_DIR = args.data_dir
print (DATA_DIR)

query = {}
query['disease']= args.disease.split(';')
query['drug']=args.drug.split(';')
query['gene']=args.gene.split(';')
query['function']= args.function.split(';')
print (query)

diffusion, diffusion_n2i, diffusion_i2n, networks, node2tp, tp2node, node2ct = read_server_data(DATA_DIR)

for tp in query:
	for w in query[tp]:
		if w not in tp2node[tp]:
			warnings.warn(w+' not in current '+tp+' list')
#python run_deepsyn_fast.py --data_dir "DeepSyn/data/knowledge_network/" --drug "doxurbicin" --disease "breast cancer;lung cancer" --gene "top2a;kmt2a" --function "dna repair" --output_file "test.txt"


ans_paths, ans_nodes = run_query(query, networks, diffusion, diffusion_n2i, diffusion_i2n, node2tp, MAX_DEPTH = 4)
print (ans_paths)
nx_obj = create_networkx_obj(ans_paths, ans_nodes, node2tp)
write_to_cyto_scape(ans_paths, ans_nodes, node2tp, args.output_file)

print (nx_obj)
