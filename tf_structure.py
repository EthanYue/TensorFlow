import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# session control
with tf.Session() as sess:
	sess.run(init)  # Very important
	for step in range(201):
		sess.run(train)
		if step % 20 == 0:
			print(step, sess.run(Weights), sess.run(biases))

# 在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output
output = tf.multiply(input1, input2)

with tf.Session() as sess:
	print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
