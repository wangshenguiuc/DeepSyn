import numpy as np
import collections
import sys

class WordNet():

	def __init__(self):
		self.word_ct = collections.defaultdict(dict)
		self.word_type = {}
		self.total_wt = {}
		nested_dict = lambda:collections.defaultdict(nested_dict)
		self.edge_ct = nested_dict()

	def ReadWordNet(self, file_name_l, verbal=False, print_every = 3000000,min_freq_cutoff=3):
		#print file_name
		for i,file_name in enumerate(file_name_l):
			fin = open(file_name)
			source = file_name_l[file_name]
			total_wt = 0.
			for ct,line in enumerate(fin):
				w = line.strip().split('\t')
				w1,c1,w2,c2,c12 = w[0:5]
				c1 = int(c1)
				c2 = int(c2)
				c12 = int(c12)
				total_wt += c12
				self.word_ct[source][w1] = self.word_ct[source].get(w1,0) + c12
				self.word_ct[source][w2] = self.word_ct[source].get(w2,0) + c12
				if c12 < min_freq_cutoff:
					continue

				self.edge_ct[source][w1][w2] = c12*1.0 / np.sqrt(c1*c2)
				self.edge_ct[source][w2][w1] = c12*1.0 / np.sqrt(c1*c2)

				if verbal and ct%print_every==0:
					output = 'finished {0}'.format(ct)
					print source,output,c12,w1,w2
					sys.stdout.flush()
			fin.close()
			self.total_wt[source] = total_wt
			#self.normalize_network_weight(source)
		return self.edge_ct,self.word_ct

	def normalize_network_weight(self,source):
		total_wt = self.total_wt[source]
		for w1 in self.edge_ct[source]:
			for w2 in self.edge_ct[source][w1]:
				self.edge_ct[source][w1][w2] /= total_wt

	def ReadNodeTypeNCBI(self,default_score=100, file_name = 'data/NLP_Dictionary/description.txt',verbal=False, print_every = 100000):
		synonym_g2g = collections.defaultdict(dict)
		description_t2g = collections.defaultdict(dict)
		fin = open(file_name)
		for ct,line in enumerate(fin):
			w = line.lower().strip().split('@')
			if len(w)>3:
				continue
			id,gl,text = w
			gl = gl.split(',')
			for g1 in gl:
				for g2 in gl:
					synonym_g2g[g1][g2] = default_score
					synonym_g2g[g2][g1] = default_score
					self.word_type[g1] = 'ncbi_gene'
					self.word_type[g2] = 'ncbi_gene'
			text = line.lower().translate(None, ',:()=%>/[]')
			sent_l = text.strip().split('.')
			sent_ct = 0
			for sent in sent_l:
				pset = parse_sentence(self.word_ct,sent)
				for w in pset:
					for g in gl:
						description_t2g[w][g] = default_score
			if verbal and ct%print_every==0:
				output = 'finished {0}'.format(ct)
				print output
		fin.close()
		return synonym_g2g

	def ReadNodeTypeLiterome(self, default_score=100, file_name = 'data/NLP_Dictionary/pubmed.network'):
		literome_g2g = collections.defaultdict(dict)
		fin = open(file_name)
		for line in fin:
			g1,g2,type,conf = line.lower().strip().split('\t')
			literome_g2g[g1][g2] = default_score
			literome_g2g[g2][g1] = default_score
			self.word_type[g1] = 'literome_gene'
			self.word_type[g2] = 'literome_gene'
		fin.close()
		return literome_g2g

	def ReadNodeTypePPI(self, default_score=100, file_name = 'data/network/human/InBio-Map_Symbol.sif'):
		PPI_g2g = collections.defaultdict(dict)
		fin = open(file_name)
		for line in fin:
			g1,g2 = line.lower().strip().split('\t')
			PPI_g2g[g1][g2] = default_score
			PPI_g2g[g2][g1] = default_score
			self.word_type[g1] = 'ppi_gene'
			self.word_type[g2] = 'ppi_gene'
		fin.close()
		return PPI_g2g

	def ReadNodeTypeAutoPhrase(self, file_name = 'data/NLP_Dictionary/AutoPhrase_multi-words.txt.only_GO_based',sc_cutoff = 0.8):
		fin = open('data/NLP_Dictionary/AutoPhrase_multi-words.txt.only_GO_based')
		for line in fin:
			w = line.lower().strip().split('\t')
			sc = float(w[0])
			if sc<sc_cutoff:
				continue
			self.word_type[w[1]] = 'Han_function'
		fin.close()

	def ReadNodeTypePercha(self, file_name = 'data/NLP_Dictionary/all_entity.txt'):
		fin = open(file_name)
		for line in fin:
			w  =line.strip().lower().split('\t')
			if w[0].replace('_',' ') in self.word_type:
				self.word_type[w[0].replace('_',' ')] ='Percha_'+w[1]
			else:
				self.word_type[w[0].replace('_',' ')] ='Percha_'+w[1]
		fin.close()
	def ReadNodeTypeSider(self, default_score=100, file_drug2adr = 'data/side_effect/bio-decagon-mono.csv'):
		hpo_d2d = collections.defaultdict(dict)
		fin = open(file_drug2adr)
		for line in fin:
			w = line.strip().lower().split(',')
			self.word_type[w[2]] = 'sider_disease'
		fin.close()

	def ReadNodeTypeHPO(self, default_score=100, file_hpo2gene = 'data/NLP_Dictionary/hpo_gene.txt', file_phenotype = 'data/NLP_Dictionary/phenotype_annotation.tab',file_obo = 'data/NLP_Dictionary/hp_obo_parsed.tsv'):
		hpo_d2d = collections.defaultdict(dict)
		fin = open(file_phenotype)
		for line in fin:
			w = line.strip().lower().split('\t')
			self.word_type[w[2]] = 'hpo_disease'
		fin.close()

		hpo_f2g = collections.defaultdict(dict)
		fin = open(file_hpo2gene)
		for line in fin:
			w = line.strip().lower().split('\t')
			d = w[1]
			g = w[3]
			hpo_f2g[d][g] = default_score
			self.word_type[d] = 'hpo_disease'
			self.word_type[g] = 'hpo_gene'
		fin.close()

		fin = open(file_obo)
		for line in fin:
			w = line.strip().lower().split('\t')
			w1 = w[0].split('(')[1].replace(')','')
			w2 = w[1].split('(')[1].replace(')','')
			self.word_type[w1] = 'hpo_disease'
			self.word_type[w2] = 'hpo_disease'
			hpo_d2d[w2][w1] = default_score
		fin.close()
		return hpo_d2d,hpo_f2g

	def ReadNodeTypeTissue(self, file_name = 'data/NLP_Dictionary/tissue.txt'):
		fin = open(file_name)
		for line in fin:
			w =line.strip().lower()
			self.word_type[w] = 'barbasi_tissue'
		fin.close()

	def ReadNodeTypeMesh(self,file_name = 'data/NLP_Dictionary/2017MeshTree.txt'):
		mesh_mapping = {'A':'tissue','B':'tissue','C':'disease','D':'drug','E':'drug','F':'disease',
					   'G':'function','H':'entity','I':'entity','J':'entity','K':'entity','M':'entity','L':'entity',
					   'N':'entity','V':'entity','Z':'entity'}
		fin = open(file_name)
		fin.readline()
		for line in fin:
			w =line.strip().lower().split('\t')
			t = mesh_mapping[w[0][0].upper()]
			self.word_type[w[2]] = 'mesh_' + t
		fin.close()

	def ReadNodeTypeGO(self,file_name = 'data/NLP_Dictionary/go_term.txt'):
		fin = open(file_name)
		for line in fin:
			w  =line.strip().lower()
			self.word_type[w] = 'GO_function'
		fin.close()

	def ReadNodeTypeCCLE(self,file_name = 'data/NLP_Dictionary/top_genes_exp_hgnc.txt'):
		fin = open(file_name)
		for line in fin:
			w = line.lower().strip().split('\t')
			self.word_type[w[0]] = 'CCLE_gene'
		fin.close()

	def ReadNodeTypeMonarchDiseaseGene(self,file_name = 'data/NLP_Dictionary/gene_disease_formatted.txt'):
		fin = open(file_name)
		d2g = collections.defaultdict(dict)
		background_gene = set()
		for line in fin:
			w = line.lower().strip().split('\t')
			d = w[0]
			g = w[1]
			#if d not in self.edge_ct:
			#	continue
			d2g[d][g] = 1
			self.word_type[g] = 'monarch_gene'
			self.word_type[d] = 'monarch_disease'
			background_gene.add(g)
		fin.close()
		return background_gene,d2g

	def ReadNodeTypeCtrpDrugGene(self,file_name = 'data/NLP_Dictionary/drug_target_formatted.txt'):
		fin = open(file_name)
		d2g = collections.defaultdict(dict)
		background_gene = set()
		for line in fin:
			w = line.lower().strip().split('\t')
			d = w[0]
			g = w[1]
			#if d not in self.edge_ct:
			#	continue
			d2g[d][g] = 1
			self.word_type[g] = 'ctrp_gene'
			self.word_type[d] = 'ctrp_drug'
			background_gene.add(g)
		fin.close()
		return background_gene,d2g

	def ReadNodeTypeCancerGene(self,file_name = 'data/NLP_Dictionary/cancer_gene.txt'):
		fin = open(file_name)
		d2g = collections.defaultdict(dict)
		background_gene = set()
		for line in fin:
			w = line.lower().strip().split('\t')
			if w[1] not in self.edge_ct:
				continue
			d2g[w[1]][w[0]] = 1
			self.word_type[w[0]] = 'cancer_gene'
			self.word_type[w[1]] = 'cancer_disease'
			background_gene.add(w[0])
		fin.close()
		return background_gene,d2g

	def ReadEdgeTypePerchar(self,kg_thres = 5000, default_score = 100,file_name = 'data/NLP_Dictionary/all_relation.txt'):
		kg_relation = collections.defaultdict(dict)
		fin = open(file_name)
		for line in fin:
			w =line.strip().lower().split('\t')
			w1 = w[0].replace('_',' ')
			w2 = w[1].replace('_',' ')
			if w1 in self.stop_word_list or w2 in self.stop_word_list:
				continue
			if float(w[3]) < kg_thres:
				continue
			if w1 not in self.word_type or w2 not in self.word_type:
				continue
			sent = w[4]
			if [self.word_type[w1].split('_')[1],self.word_type[w2].split('_')[1]] in self.graph_edge_type:
				kg_relation[w1][w2] = default_score
			if [self.word_type[w2].split('_')[1],self.word_type[w1].split('_')[1]] in self.graph_edge_type:
				kg_relation[w2][w1] = default_score
		fin.close()
		return kg_relation

	def ReadEdgeTypeGO(self,default_score = 100,file_name = 'data/NLP_Dictionary/GO_term.network'):
		go_f2f = collections.defaultdict(dict)
		fin = open(file_name)
		for line in fin:
			w1,w2 =line.strip().lower().split('\t')[0:2]
			go_f2f[w1][w2] = default_score
			#go_f2f[w2][w1] = default_score
		fin.close()
		return go_f2f

	def ReadEdgeTypeGO2Gene(self,default_score = 100,file_name = 'data/NLP_Dictionary/go2gene_set_complete.txt'):
		go_f2g = collections.defaultdict(dict)
		fin = open(file_name)
		for line in fin:
			w = line.strip().lower().split('\t')
			for i in range(1, len(w)):
				go_f2g[w[0]][w[i]] = default_score
		fin.close()
		return go_f2g

	def ReadWordType(self,default_score = 100000,use_auto_phrase=False):
		if use_auto_phrase:
			self.ReadNodeTypeAutoPhrase()
		#self.ReadNodeTypePercha()
		self.literome_g2g = self.ReadNodeTypeLiterome(default_score=default_score)
		self.PPI_g2g = self.ReadNodeTypePPI(default_score=default_score)
		self.synonym_g2g = self.ReadNodeTypeNCBI(default_score=default_score)
		self.hpo_d2d, self.hpo_f2g = self.ReadNodeTypeHPO(default_score=default_score)
		self.Monarch_d2g = self.ReadNodeTypeMonarchDiseaseGene()[1]
		self.ReadNodeTypeTissue()
		self.ReadNodeTypeMesh()
		self.ReadNodeTypeCtrpDrugGene(file_name = 'data/drug/gdsc/drug_target_mapped.txt')
		self.ReadNodeTypeCtrpDrugGene(file_name = 'data/drug/ctrp/drug_target_formatted.txt')
		self.ReadNodeTypeGO()
		self.ReadNodeTypeCCLE()
		self.ReadNodeTypeSider()
		self.candidate_gene,self.cancer_d2g = self.ReadNodeTypeCancerGene()
		print 'finished'
		#return self.word_type

	def ReadEdgeType(self,stop_word_list,edge_list_l=[],default_score = 100000):
		self.stop_word_list = stop_word_list
		self.graph_edge_type = []


		#'hpo_disease','Percha_disease','cancer_disease',
		# 'mesh_tissue','barbasi_tissue',
		#'Percha_drug','mesh_drug'
		#'GO_function', 'Han_function','mesh_entity', 'mesh_function'
		#'CCLE_gene', 'cancer_gene',  'literome_gene', 'mesh_disease','ncbi_gene','Percha_gene'
		if len(edge_list_l) == 0:
			self.graph_edge_type.append(['disease','symptom'])
			self.graph_edge_type.append(['disease','tissue'])
			self.graph_edge_type.append(['disease','function'])
			#self.graph_edge_type.append(['disease','disease'])
			#self.graph_edge_type.append(['disease','gene'])

			self.graph_edge_type.append(['tissue','function'])
			#self.graph_edge_type.append(['tissue','gene'])
			self.graph_edge_type.append(['tissue','symptom'])
			#self.graph_edge_type.append(['tissue','tissue'])

			self.graph_edge_type.append(['symptom','function'])
			self.graph_edge_type.append(['symptom','tissue'])
			#self.graph_edge_type.append(['symptom','symptom'])
			#self.graph_edge_type.append(['symptom','gene'])

			#edge_type.append(['function','gene'])
			self.graph_edge_type.append(['function','function'])
			self.graph_edge_type.append(['function','gene'])
			#self.graph_edge_type.append(['gene','gene'])
		else:
			for i in edge_list_l:
				self.graph_edge_type.append(i)

		self.go_f2f = self.ReadEdgeTypeGO(default_score=default_score)
		self.go_f2g = self.ReadEdgeTypeGO2Gene(default_score=default_score)
		self.Perchar_kg = self.ReadEdgeTypePerchar(default_score=default_score)

def parse_sentence(word_ct,sent,max_phrase_length = 5, stop_words = [], move_sub_string=True):
	pset = {}
	wl = sent.split(' ')
	w = ''
	if len(word_ct) == 0:
		return pset
	for i in range(len(wl)):
		tmp_w = ''
		for k in range(max_phrase_length):
			if i+k >= len(wl):
				break
			tmp_w += wl[i+k]
			if tmp_w in word_ct and tmp_w not in stop_words:
				pset[tmp_w] = i
			tmp_w += ' '
	if move_sub_string:
		for ti in pset.keys():
			for tj in pset.keys():
				if ti in tj and ti!=tj:
					pset.pop(ti, None)
	return pset
