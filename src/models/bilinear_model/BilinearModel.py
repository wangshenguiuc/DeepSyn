import numpy as np

class BinearModel():

	def __init__(self, X_feat, Y_feat, X2Y):
		self.nsample,self.sample_dim = np.shape(X_feat)
		self.nlabel,self.label_dim = np.shape(Y_feat)
		self.estimate(X_feat,Y_feat,X2Y)

	def estimate(self,X_feat,Y_feat,X2Y):
		xy = np.zeros((self.nsample, self.nlabel))
		for i in range(self.nlabel):
			p_l = np.where(X2Y[:,i]==1)[0]
			n_l = np.where(X2Y[:,i]==0)[0]
			#print p_l,len(p_l)
			#print n_l,len(n_l)
			#for p in p_l:
			xy[p_l,i] = len(n_l)
			#for n in n_l:
			xy[n_l,i] = -1 * len(p_l)
		self.w = np.dot(X_feat.T, xy)
		self.w = np.dot(self.w, Y_feat)

	def predict(self,X_test_feat,Y_feat):
		Y = np.dot(X_test_feat, self.w)
		Y = np.dot(Y, Y_feat.T)
		return Y


'''
  nlabel = size(LY,1);
    nnode = size(LX,1);


    xy=zeros(nnode,nlabel);

    for i=1:nlabel
        p = intersect(find(A(:,i)==1),train_ind);
        n = intersect(find(A(:,i)==0),train_ind);
        xy(p,i)=length(n);
        xy(n,i)=-length(p);
    end

    w = LX'*xy*LY;
end
'''
