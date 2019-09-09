import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class LSTM(nn.Module):
	def __init__(self,  args=None):
		super(LSTM, self).__init__()

		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table

		"""
		self.args = args

		self.batch_size = args.batch_size
		self.output_size = args.class_num
		self.hidden_size = args.hidden_size
		self.vocab_size = args.embed_num
		self.embedding_length = args.embed_dim

		self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_length)# Initializing the look-up table.
		if hasattr(args, 'emb_weights'):
			self.emb_weights = args.emb_weights
			self.word_embeddings.weight = nn.Parameter(self.emb_weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
		self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)
		self.label = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, x, batch_size=None):

		"""
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

		Returns
		-------
		Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
		final_output.shape = (batch_size, output_size)

		"""

		''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
		cur_batch_size = x.size()[0]
		input = self.word_embeddings(x) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
		if batch_size is None:
			h_0 = Variable(torch.zeros(1, cur_batch_size, self.hidden_size).cuda()) # Initial hidden state of the LSTM
			c_0 = Variable(torch.zeros(1, cur_batch_size, self.hidden_size).cuda()) # Initial cell state of the LSTM
		else:
			h_0 = Variable(torch.zeros(1, cur_batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(1, cur_batch_size, self.hidden_size).cuda())
		output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
		final_output = self.label(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)

		return final_output
