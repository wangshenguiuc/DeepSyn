import collections
import operator
import scipy.stats as stats
import numpy as np
class PathwayEnrichment():

	def __init__(self,pathfile_l=['data/pathway/cancer_gene_set.txt',
	'data/pathway/nci_pathway_hgnc.txt'],gene_set = []):
		self.p2g = collections.defaultdict(set)
		if len(gene_set) > 0:
			self.gene_set = gene_set
		else:
			self.gene_set = set()
		for pathfile in pathfile_l:
			fin = open(pathfile)
			for line in fin:
				w = line.strip().split('\t')

				if len(gene_set) == 0:
					self.gene_set.add(w[1])
				if w[1] not in self.gene_set:
					continue
				self.p2g[w[0]].add(w[1])
			fin.close()

	def enrich(self,gene_set,ngene = 18362,thres=0.05,correction=True):
		gene_set = set(gene_set)
		tab = np.zeros((2,2))
		sign_p = {}
		sign_p_gset = {}
		best_p = None
		for p in self.p2g:
			overlap = gene_set.intersection(self.p2g[p])
			tab[1,1] = len(overlap)
			tab[1,0] = len(gene_set) - len(overlap)
			tab[0,1] = len(self.p2g[p]) - len(overlap)
			tab[0,0] = ngene - tab[1,1] - tab[0,1] - tab[1,0]
			#print tab,len(self.p2g[p])

			odds,pv = stats.fisher_exact(tab,alternative='greater')
			if correction:
				pv *= len(self.p2g)
			if pv < thres:
				sign_p[p] = pv
				sign_p_gset[p] = overlap
		plist = sorted(sign_p.items(),key=operator.itemgetter(1))
		if len(plist)>0:
			best_p = plist[0][0]
		return sign_p,sign_p_gset,best_p
