import sys
import os
repo_dir = '/cellar/users/majianzhu/Data/wangsheng/NetAnt/'
sys.path.append(repo_dir)
os.chdir(repo_dir)
import cPickle as pickle
import collections

DATA_DIR = '/data/cellardata/users/netant/Data/'
CACHE_DIR = '/data/cellardata/users/netant/Cache/'

def get_pvalue_path(DATA_DIR, function):
	pvalue_dir = DATA_DIR + '/data/pvalue/preprocess/' + function[0:2] + '/'
	if not os.path.exists(pvalue_dir):
		os.makedirs(pvalue_dir)
	return pvalue_dir + function
'''
function_score_file = DATA_DIR + '/data/pvalue/function_score/phrase/2_0.01/all_new.txt'
f2g_sc =  collections.defaultdict(dict)
fin = open(function_score_file)
for ct,line in enumerate(fin):
	if ct%200==0:
		print ct
	w = line.strip().split('\t')
	d = w[0]
	fout = open(get_pvalue_path(DATA_DIR, d),'w')
	for i in range(1,len(w)):
		fout.write(str(w[i])+'\t')
	fout.write('\n')
	fout.close()
fin.close()
'''


	
	
