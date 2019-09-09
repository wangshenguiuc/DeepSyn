import numpy as np
from sklearn.metrics import roc_auc_score
class MF():

	def __init__(self, R, K, alpha, beta, iterations):
		"""
		Perform matrix factorization to predict empty
		entries in a matrix.

		Arguments
		- R (ndarray)   : user-item rating matrix
		- K (int)	   : number of latent dimensions
		- alpha (float) : learning rate
		- beta (float)  : regularization parameter
		"""

		self.R = R
		self.num_users, self.num_items = R.shape
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.iterations = iterations

	def train(self,truth=[]):
		# Initialize user and item latent feature matrice
		self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
		self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

		# Initialize the biases
		self.b_u = np.zeros(self.num_users)
		self.b_i = np.zeros(self.num_items)
		self.b = np.mean(self.R[np.where(self.R != 0)])

		# Create a list of training samples
		self.samples = [
			(i, j, self.R[i, j])
			for i in range(self.num_users)
			for j in range(self.num_items)
			if self.R[i, j] > 0
		]
		print len(self.samples)
		# Perform stochastic gradient descent for number of iterations
		training_process = []
		for i in range(self.iterations):
			np.random.shuffle(self.samples)
			self.sgd()
			mse = self.mse()
			training_process.append((i, mse))
			if (i+1) % 1 == 0:
				all_mat = self.full_matrix()
				print np.min(all_mat),np.max(all_mat)
				our_auc,our_all_auc = self.evaluate_matrix(truth,all_mat)
				print("Iteration: %d ; error = %.4f; auc=%f, all_auc=%f" % (i+1, mse, our_auc, our_all_auc))

		return training_process

	def mse(self):
		"""
		A function to compute the total mean square error
		"""
		xs, ys = self.R.nonzero()
		predicted = self.full_matrix()
		error = 0
		for x, y in zip(xs, ys):
			error += pow(self.R[x, y] - predicted[x, y], 2)
		return np.sqrt(error)

	def sgd(self):
		"""
		Perform stochastic graident descent
		"""
		for i, j, r in self.samples:
			# Computer prediction and error
			prediction = self.get_rating(i, j)
			e = (r - prediction)

			# Update biases
			self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
			self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

			# Update user and item latent feature matrices
			self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
			self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

	def get_rating(self, i, j):
		"""
		Get the predicted rating of user i and item j
		"""
		prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
		return prediction

	def full_matrix(self):
		"""
		Computer the full matrix using the resultant biases, P and Q
		"""
		return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

	def evaluate_matrix(self, truth,pred,bin_val=0.8):
		ns = np.shape(truth)[0]
		auc_l = []
		all_auc_l = []
		for i in range(ns):
			#spear_l.append( stats.spearmanr(pred[i,:],truth[i,:])[0])
			bin =  [1.0 if x >= bin_val else 0.0 for x in truth[i]]
			ind_list = range(0,i)
			ind_list.extend( range(i+1,ns))
			bin = np.array(bin)
			ind_list = np.array(ind_list)
			if np.sum(bin[ind_list])==0:
				continue
			auc = roc_auc_score(bin[ind_list], pred[i,ind_list])
			auc_l.append(auc)

			bin =  [1.0 if x >= bin_val else 0.0 for x in truth[i]]
			all_auc = roc_auc_score(bin, pred[i,:])
			all_auc_l.append(all_auc)
		return np.mean(auc_l),np.mean(all_auc_l)
