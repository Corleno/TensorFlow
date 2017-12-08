### Numpy
import numpy as np

a = np.zeros((2,2)); b = np.ones((2,2))
print (np.sum(b, axis=1))

a.shape

print (np.reshape(a, (1,4)))

### TensorFlow
import tensorflow as tf

tf.InteractiveSession()

a = tf.zeros((2,2)); b = tf.ones((2,2))
print (tf.reduce_sum(b, reduction_indices=1).eval())

a.get_shape()

print (tf.reshape(a, (1,4)).eval())

W1 = tf.Variable(tf.zeros((2,2)), name = "weights")

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print (sess.run(W1))