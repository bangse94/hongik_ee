#full adder use ReLU function

import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

lr = 0.1

#trainning data & placeholder
input_data = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]], dtype=np.float32)
output_data = np.array([[0,0],[0,1],[0,1],[1,0],[0,1],[1,0],[1,0],[1,1]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#layer 1, 3 input 4 output
W1 = tf.Variable(tf.random_normal([3, 4]), name="weight1")
b1 = tf.Variable(tf.random_normal([4]), name="bias1")
layer1 = tf.nn.leaky_relu(tf.matmul(X, W1) + b1)

#layer 2, 4 input 2 output
W2 = tf.Variable(tf.random_normal([4, 2]), name="weight2")
b2 = tf.Variable(tf.random_normal([2]), name="bias2")
hypothesis = tf.nn.leaky_relu(tf.matmul(layer1, W2) + b2)

# cost function / minimize cost
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# predicate / accuracy
predicated = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicated, Y), dtype=tf.float32))

#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        sess.run(train, feed_dict={X: input_data, Y: output_data})
        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={X: input_data, Y: output_data}), sess.run([W1, W2]))
    h, c, a = sess.run([hypothesis, predicated, accuracy], feed_dict={X: input_data, Y: output_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
