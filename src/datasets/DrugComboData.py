import numpy as np
import pandas as PD
import collections
import csv




class DrugComboData():
	def __init__(self, gene_list = [], dataset_path = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/drug/dream_drug_combo/raw_data/dcombi/', select_drug=[],
	drug_auc_file_l = ['ch2_LB.csv','ch1_LB.csv','Drug_synergy_data/ch1_train_combination_and_monoTherapy.csv'],
	tf = 'Drug_synergy_data/Drug_info_release.csv',format_string=False):
		self.combo2c = collections.defaultdict(dict)
		self.d2c = collections.defaultdict(dict)
		self.d2g = collections.defaultdict(dict)
		self.drug_cl_set = set()
		for df in drug_auc_file_l:
			drug_auc_file = dataset_path + df
			fin = open(drug_auc_file)
			all_data = list(csv.reader(fin, delimiter=','))
			head = all_data[0]
			the_data = all_data[1:]
			#head = fin.readline().strip().split(',')
			cid = head.index('CELL_LINE')
			did1 = head.index('COMPOUND_A')
			did2 = head.index('COMPOUND_B')
			scid = head.index('SYNERGY_SCORE')
			sc1id = head.index('IC50_A')
			sc2id = head.index('IC50_B')
			for line in the_data:
				#w = line.strip().split(',')
				w = line
				c = w[cid]
				d1 = w[did1]
				d2 = w[did2]
				sc = float(w[scid])
				sc1 = float(w[sc1id])
				sc2 = float(w[sc2id])
				self.drug_cl_set.add(c)
				self.combo2c[d1+'.'+d2][c] = sc
				if d1 not in self.d2c or c not in self.d2c[d1]:
					self.d2c[d1][c] = []
				if d2 not in self.d2c or c not in self.d2c[d2]:
					self.d2c[d2][c] = []
				self.d2c[d1][c].append(sc1)
				self.d2c[d2][c].append(sc2)
			fin.close()

		drug_target_file = dataset_path + tf
		fin = open(drug_target_file)
		all_data = list(csv.reader(fin, delimiter=','))
		head = all_data[0]
		the_data = all_data[1:]
		tid = head.index('Target(Official Symbol)')
		did = head.index('ChallengeName')

		fin.close()

		for line in the_data:
			w = line
			dl = w[tid].replace(' ','')
			t = w[did]
			#"AKT*,SGK*"
			if '"' in dl:
				dl = dl.replace('"','')
			dl = dl.strip().split(',')
			for d in dl:
				if '*' in d:
					gl = self.map_gene_reg(d,gene_list)
					for g in gl:
						self.d2g[t][g] = 1
				else:
					self.d2g[t][d] = 1


	def map_gene_reg(self, g,gene_list):
		a = []
		gprefix = g.replace('*','')
		for g in gene_list:
			if g.startswith(gprefix):
				a.append(g)
		return a
