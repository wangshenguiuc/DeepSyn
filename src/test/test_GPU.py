import os
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import time
import numpy as np

n = 8192

def mat_multiply(A, B):
	config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	with tf.device("/gpu:0"):
		matrix1 =  tf.constant(A, dtype=tf.float64)
		matrix2 =  tf.constant(B, dtype=tf.float64)
		product = tf.matmul(matrix1, matrix2)
	ans = sess.run(product)
	return ans

# avoid optimizing away redundant nodes

iters = 10

# pre-warming
A = np.ones((n,n))
B = np.ones((n,n))
start = time.time()
for i in range(1):
	ans = mat_multiply(A,B)
end = time.time()
ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications
elapsed = (end - start)
rate = iters*ops/elapsed/10**9
print('\n %d x %d matmul took: %.2f sec, %.2f G ops/sec' % (n, n,
                                                            elapsed/iters,
                                                            rate,))

start = time.time()
a = np.ones((n,n))
b = np.ones((n,n))
for i in range(1):
	np.dot(a,b)
end = time.time()
ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications
elapsed = (end - start)
rate = iters*ops/elapsed/10**9
print('\n %d x %d matmul took: %.2f sec, %.2f G ops/sec' % (n, n,
                                                            elapsed/iters,
                                                            rate,))
