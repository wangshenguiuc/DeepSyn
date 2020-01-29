import sys
import os
from utils import *
DATA_DIR = 'DeepSyn/data/knowledge_network/'
print (DATA_DIR)
diffusion, diffusion_n2i, diffusion_i2n, networks, node2tp, tp2node, node2ct = read_server_data(DATA_DIR)

query = {}
query['disease']= ['breast cancer']
query['drug']=[]
query['gene']=['top2a']
query['function']= []

ans_paths, ans_nodes = run_query(query, networks, diffusion, diffusion_n2i, diffusion_i2n, node2tp, MAX_DEPTH = 4)

print (ans_paths)
