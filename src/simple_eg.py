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

### Tensorflow variable
W1 = tf.Variable(tf.zeros((2,2)), name = "weights")

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print (sess.run(W1))

### add and multiple
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(1.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)
with tf.Session() as sess:
	result = sess.run([mul, intermed])
	print (result)

### inputting Data
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
	print(sess.run(ta))

### Placeholders
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
	print (sess.run([output], feed_dict={input1:[7.0], input2:[2.0]}))

### Variable Scope
with tf.variable_scope("foo"):
	v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse = True):
	v1 = tf.get_variable("v", [1])
assert v1 == v

### Linear Regression in TensorFlow
import numpy as np
import seaborn

# Define input data
X_data = np.arange(100, step=.1)
y_data = X_data + 20*np.sin(X_data/10)

# Plot input data
# import matplotlib.pyplot as plt
# plt.scatter(X_data, y_data)
# plt.show()

# Define data size and batch size
n_samples = 1000
batch_size = 100

# Tensorflow is finicky about shapes, so resize
X_data = np.reshape(X_data, (n_samples,1))
y_data = np.reshape(y_data, (n_samples,1))

# Define placeholders for input
X = tf.placeholder(tf.float32, shape = (batch_size, 1))
y = tf.placeholder(tf.float32, shape = (batch_size, 1))

# Define variables to be learned
with tf.variable_scope("linear-regression"):
	W = tf.get_variable("weights", (1, 1),
		initializer = tf.random_normal_initializer())
	b = tf.get_variable("bias", (1,),
		initializer = tf.constant_initializer(0.0))
	y_pred = tf.matmul(X, W) + b
	loss = tf.reduce_sum((y - y_pred)**2/n_samples)

# Sample code to run one step of gradient desent
opt = tf.train.AdamOptimizer()
opt_operation = opt.minimize(loss)

with tf.Session() as sess:
	# Initializer Variables in graph
	sess.run(tf.initialize_all_variables())
	# Gradient descent loop for 500 steps
	for _ in range(500):
		# Select random minibatch
		indices = np.random.choice(n_samples, batch_size)
		X_batch, y_batch = X_data[indices], y_data[indices]
		# Do gradient descent step
		_, loss_val = sess.run([opt_operation, loss], feed_dict={X: X_batch, y: y_batch})
		print (loss_val)

#