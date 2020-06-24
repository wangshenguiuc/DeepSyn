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

try:
	diffusion, diffusion_n2i, diffusion_i2n, networks, node2tp, tp2node, node2ct = read_server_data(DATA_DIR)
	node_info, term2pid = read_node_info(DATA_DIR)
except Exception as e:
	exit('load database error '+str(e))

try:
	query = {}
	query['disease']= args.disease.lower().split(';')
	query['drug']=args.drug.lower().split(';')
	query['gene']=args.gene.lower().split(';')
	query['function']= args.function.lower().split(';')
	if not valid_query(query, tp2node):
		raise ValueError('query cannot be empty')
	query = remove_non_db_terms(query, tp2node)
	print (query)
except Exception as e:
	exit('Query error '+str(e))

#python run_deepsyn_fast.py --data_dir "DeepSyn/data/knowledge_network/" --drug "doxurbicin" --disease "breast cancer;lung cancer" --gene "top2a;kmt2a" --function "dna repair" --output_file "test.txt"

try:
	ans_paths, ans_nodes = run_query(query, networks, diffusion, diffusion_n2i, diffusion_i2n, node2tp, MAX_DEPTH = 4)
except Exception as e:
	exit('run query error '+str(e))


try:
	for node in ans_nodes:
		node_info, title, description, url = query_node(node, node_info, term2pid, node2tp, DATA_DIR)
		print (node, title, description, url)
	for path in ans_paths:
		path_info = query_edge(path, DATA_DIR)
		print (path, path_info)
except Exception as e:
	exit('retrieve node info error '+str(e))

try:
	nx_obj = create_networkx_obj(ans_paths, ans_nodes, node2tp)
	write_to_cyto_scape(ans_paths, ans_nodes, node2tp, args.output_file)
	print (nx_obj)
except Exception as e:
	exit('generate networkx file error '+str(e))
