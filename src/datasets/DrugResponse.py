import numpy as np
import pandas as PD
import collections



class DrugResponse():
	def __init__(self, dataset_path, select_drug=[],drug_auc_file = 'label_mapped.txt',drug_target_file = 'drug_target_mapped.txt',format_string=False):
		self.d2c = collections.defaultdict(dict)
		self.d2c_norm = collections.defaultdict(dict)
		self.d2g = collections.defaultdict(dict)
		drug_auc_file = dataset_path + drug_auc_file
		drug_target_file = dataset_path + drug_target_file
		self.drug_cl_set = set()
		max_d2c = -1
		with open(drug_auc_file) as f:
			for line in f:
				w = line.strip().upper().split('\t')
				c = w[1]
				d = w[0]
				if format_string:
					d = d.replace('-','')
				if len(select_drug) > 0 and d not in select_drug:
					continue
				self.d2c[d][c] = float(w[2])
				max_d2c = max(max_d2c, float(w[2]))
				self.drug_cl_set.add(c)
		for d in self.d2c:
			for c in self.d2c[d]:
				self.d2c_norm[d][c] = self.d2c[d][c] / max_d2c

		with open(drug_target_file) as f:
			for line in f:
				w = line.upper().strip().split('\t')
				d = w[0]
				if format_string:
					d = d.replace('-','')
				if len(select_drug) > 0 and d not in select_drug:
					continue
				if len(w) == 1:
					continue
				tgt = w[1].split(';')
				for t in tgt:
					self.d2g[d][t] = 1
