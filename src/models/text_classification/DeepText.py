from torchtext import data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import torch.autograd as autograd
import sys
import os
import torch.cuda as cutorch
from torchtext.vocab import Vectors, GloVe
from sklearn import metrics
import GPUtil

class MR(data.Dataset):

	@staticmethod
	def sort_key(ex):
		return len(ex.text)

	def __init__(self, text_field, label_field, trainX= None, trainY = None, examples=None, **kwargs):
		"""Create an MR dataset instance given a path and fields.
		Arguments:
			text_field: The field that will be used for text data.
			label_field: The field that will be used for label data.
			path: Path to the data file.
			examples: The examples contain all the data.
			Remaining keyword arguments: Passed to the constructor of
				data.Dataset.
		"""

		#text_field.preprocessing = data.Pipeline(clean_str)

		fields = [('text', text_field), ('label', label_field)]
		if examples is None:
			examples = []
			for i in range(len(trainX)):
				examples.append(data.Example.fromlist([trainX[i], trainY[i]], fields))

		super(MR, self).__init__(examples, fields, **kwargs)

	@classmethod
	def splits(cls, text_field, label_field, trainX, trainY, dev_ratio=.1, shuffle=True, root='.', **kwargs):
		"""Create dataset objects for splits of the MR dataset.
		Arguments:
			text_field: The field that will be used for the sentence.
			label_field: The field that will be used for label data.
			dev_ratio: The ratio that will be used to get split validation dataset.
			shuffle: Whether to shuffle the data before split.
			root: The root directory that the dataset's zip archive will be
				expanded into; therefore the directory in whose trees
				subdirectory the data files will be stored.
			train: The filename of the train data. Default: 'train.txt'.
			Remaining keyword arguments: Passed to the splits method of
				Dataset.
		"""

		examples = cls(text_field, label_field,trainX=trainX, trainY=trainY, **kwargs).examples

		if shuffle: random.shuffle(examples)
		dev_index = -1 * int(dev_ratio*len(examples))

		return (cls(text_field, label_field, examples=examples[:dev_index]),
				cls(text_field, label_field, examples=examples[dev_index:]))


# load SST dataset
def sst(text_field, label_field,  **kargs):
	train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
	text_field.build_vocab(train_data, dev_data, test_data)
	label_field.build_vocab(train_data, dev_data, test_data)
	train_iter, dev_iter, test_iter = data.BucketIterator.splits(
										(train_data, dev_data, test_data),
										batch_sizes=(batch_size,
													 len(dev_data),
													 len(test_data)),
										**kargs)
	return train_iter, dev_iter, test_iter


class DeepText():

	def __init__(self,working_dir,DeepModel,use_glove = False,min_freq=10,hidden_size=256,batch_size=32,embed_dim=50,dropout=0.5,lr=0.001,kernel_num=100,epochs=100,min_epoch=50,early_stop=20):
		self.kernel_sizes = '3,4,5'
		self.kernel_num = kernel_num
		self.embed_dim = embed_dim
		self.dropout = dropout
		self.lr = lr#0.001
		self.epochs = epochs
		self.min_epoch = min_epoch
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.log_interval = 1
		self.test_interval = 2
		self.save_interval = 500
		self.early_stop = early_stop
		self.save_best = True
		self.save_dir = working_dir + '/result/TextClassify/'
		self.static = False
		self.no_cuda = False
		self.min_freq = min_freq
		self.use_glove = use_glove
		self.text_field = data.Field(lower = True)
		self.label_field = data.Field(sequential=False, unk_token = None)
		self.DeepModel = DeepModel

		#self.init_model()

	def fit(self, trainX, trainY):
		train_data, dev_data = MR.splits(self.text_field, self.label_field, trainX = trainX, trainY = trainY)
		if self.use_glove:
			self.text_field.build_vocab(train_data, dev_data, vectors=GloVe(name='6B', dim=self.embed_dim ,cache='data/network/embedding/glove/'), min_freq=self.min_freq)
			self.emb_weights = self.text_field.vocab.vectors
		else:
			self.text_field.build_vocab(train_data, dev_data, min_freq = self.min_freq)
		self.label_field.build_vocab(train_data, dev_data)
		#print self.label_field.vocab.itos
		#print self.label_field.vocab.stoi
		self.train_iter, self.dev_iter = data.Iterator.splits(
									(train_data, dev_data),
									batch_sizes=(64, 64),device=-1, repeat=False
									)
		#train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)

		# update args and print
		self.embed_num = len(self.text_field.vocab)
		self.class_num = len(self.label_field.vocab)
		#print train_data
		#print np.shape(train_data)
		#sys.exit(-1)
		print self.embed_num,self.class_num,len(train_data),len(dev_data)
		sys.stdout.flush()
		#sys.exit()
		#print 'embe_num',self.embed_num,'class_num',self.class_num
		self.cuda = (not self.no_cuda) and torch.cuda.is_available(); del self.no_cuda
		self.kernel_sizes = [int(k) for k in self.kernel_sizes.split(',')]


		# model
		self.model = self.DeepModel(args=self)

		if self.cuda:
			torch.cuda.set_device(-1)
			self.model  = self.model.cuda()

		self.train(self.train_iter, self.dev_iter, self.model , self)

	def clip_gradient(self, model, clip_value):
	    params = list(filter(lambda p: p.grad is not None, model.parameters()))
	    for p in params:
	        p.grad.data.clamp_(-clip_value, clip_value)



	def train(self, train_iter, dev_iter, model, args):
		if args.cuda:
			model.cuda()

		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		early_stop = False
		steps = 0
		best_auc = 0
		last_step = 0
		model.train()
		for epoch in range(1, args.epochs+1):
			#print epoch,args.epochs
			avg_loss = 0
			ct = 0.

			steps += 1
			for bi,batch in enumerate(train_iter):
				feature, target = batch.text, batch.label
				feature.data.t_()
				#target.data.sub_(1)  # batch first, index align
				if args.cuda:
					torch.cuda.empty_cache()
					feature, target = feature.cuda(), target.cuda()
				optimizer.zero_grad()
				logit = model(feature)
				#print bi,target.data,logit
				#print('logit vector', logit.size())
				#print('target vector', target.size())
				loss = F.cross_entropy(logit, target)
				loss.backward()
				#self.clip_gradient(model, 1e-1)
				optimizer.step()
				avg_loss = loss.item()
				ct += 1
				if steps % args.log_interval == 0 and bi==1 and epoch %10==0:
					#print torch.max(logit, 1)[1],target.data

					#corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
					logit_np = logit.data.cpu().numpy()
					target_np = target.data.cpu().numpy()
					auc = self.get_auc(logit_np, target_np)
					#accuracy = 100.0 * corrects/batch.batch_size
					size = batch.batch_size
					'''
					print('before Evaluation batch:{} - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(bi, avg_loss,accuracy, corrects,size))

					logit = model(feature)
					loss = F.cross_entropy(logit, target)
					avg_loss = loss.data[0]
					corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
					accuracy = 100.0 * corrects/batch.batch_size
					print('{} after Evaluation batch:{} - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(epoch, bi, avg_loss,accuracy, corrects,size))
					'''
				if bi==len(train_iter)-1:
					train_auc = self.eval(train_iter, model, args, epoch=epoch)
					dev_auc = self.eval(dev_iter, model, args, epoch=epoch)
					model.train()
					if dev_auc > best_auc:
						best_auc = dev_auc
						last_step = steps
						if args.save_best:
							self.save(model, args.save_dir, 'best', steps)
					else:
						if epoch - last_step >= args.early_stop and epoch > args.min_epoch:
							early_stop = True
							break
							#print('early stop by {} steps.'.format(args.early_stop))
				elif steps % args.save_interval == 0:
					self.save(model, args.save_dir, 'snapshot', steps)
			if early_stop:
				break


	def eval(self,data_iter, model, args, epoch=0):
		model.eval()
		corrects, avg_loss = 0, 0
		target_l = []
		logit_l = []
		for batch in data_iter:
			feature, target = batch.text, batch.label
			feature.data.t_()
			#target.data.sub_(1)  # batch first, index align
			if args.cuda:
				#print 'eval step',feature.size()
				torch.cuda.empty_cache()
				#print GPUtil.showUtilization()
				feature, target = feature.cuda(), target.cuda()


			logit = model(feature)
			loss = F.cross_entropy(logit, target, size_average=False)

			avg_loss += loss.item()
			corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

			logit = logit.data.cpu().numpy()
			target = target.data.cpu().numpy()
			logit_l.append(logit)
			target_l.append(target)
		#target = torch.cat(target_l,dim=0)
		#logit = torch.cat(logit_l, dim=0)
		target_l = np.array(target_l)
		logit_l = np.array(logit_l)
		#print np.shape(logit_l),np.shape(target_l)
		auc = self.get_auc(logit, target)
		size = len(data_iter.dataset)
		avg_loss /= size
		accuracy = 100.0 * corrects.item()/size
		print('{} Evaluation - loss: {:.6f}  auc: {:.4f}%({}/{})'.format(epoch, avg_loss,auc, corrects,size))
		return accuracy

	def get_auc(self,logit,target,pos_label=1):
		#output_val = logit.data.cpu().numpy()
		output_val = logit
		#truth_l = target.data.cpu().numpy()
		truth_l = target
		nsample = np.shape(output_val)[0]
		prob_l = []
		for ii in range(nsample):
			prob_l.append(np.exp(output_val[ii,pos_label]) / (np.exp(output_val[ii,0]) + np.exp(output_val[ii,1])))

		fpr,tpr,ths = metrics.roc_curve(truth_l, prob_l, pos_label = pos_label)
		auc = metrics.auc(fpr, tpr)
		return auc

	def Deep_predict(self, text, model, text_field, label_feild, cuda_flag):
		assert isinstance(text, str)
		model.eval()
		#text = text_field.tokenize(text)
		text = text_field.preprocess(text)
		text = [[text_field.vocab.stoi[x] for x in text]]
		x = text_field.tensor_type(text)
		x = autograd.Variable(x, volatile=True)
		#print 'test step',x.size()
		if cuda_flag:
			x = x.cuda()

		#print(x)
		output = model(x)
		output_val = output.data.cpu().numpy()[0]

		_, predicted = torch.max(output, 1)
		lb = predicted.data[0]

		sum = 0.
		for i in output_val:
			sum += np.exp(i)
		prob  = np.zeros(self.class_num)
		for i in range(self.class_num):
			prob[i] = np.exp(output_val[label_feild.vocab.stoi[i]]) / sum
		prob = prob / sum
		#return label_feild.vocab.itos[predicted.data[0][0]+1]
		#print prob, output_val, lb, self.label_field.vocab.stoi

		return label_feild.vocab.itos[lb],prob


	def save(self, model, save_dir, save_prefix, steps):
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		save_prefix = os.path.join(save_dir, save_prefix)
		save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
		torch.save(model.state_dict(), save_path)

	def predict_proba(self, sent):
		# train or predict
		prob = self.Deep_predict(sent[0], self.model,self.text_field, self.label_field, self.cuda)[1]
		#print('\n[Text]  {}\n[Label] {}\n'.format(predict, int(label)))
		return [prob]

	def predict(self, sent):
		# train or predict
		label = self.Deep_predict(sent[0], self.model,self.text_field, self.label_field, self.cuda)[0]
		#print('\n[Text]  {}\n[Label] {}\n'.format(predict, int(label)))
		return [label]
