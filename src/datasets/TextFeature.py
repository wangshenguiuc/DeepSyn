import numpy as np
import collections
from src.datasets.WordNet import parse_sentence

class TextFeature():

	def __init__(self):
		self.word_ct = {}

	def VecFeature(self, sent, stop_words = []):
		sent_l = line.lower().translate(None,',?!:()=%>/[]').strip().strip('.').split('. ')
        for sent in sent_l:
            pset = parse_sentence(GO_term,sent,max_phrase_length = 5, stop_words = stop_words)

