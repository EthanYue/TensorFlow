import tensorflow as tf
import numpy as np
import theano
import theano.tensor as T
import pickle


def compute_accuracy(y_target, y_predict):
	correct_prediction = np.equal(y_predict, y_target)
	accuracy = np.sum(correct_prediction) / len(correct_prediction)
	return accuracy


rng = np.random

np.random.seed(100)

N = 400
feats = 784

# generate a dataset: D = (input, target)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

# declare theano symbolic variable
x = T.dmatrix('x')
y = T.dvector('y')

# initialize the weights and biases
w = theano.shared(rng.randn(feats), name='w')
b = theano.shared(0., name='b')

# construct theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
prediction = p_1 > 0.5
xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)
cost = xent.mean() + 0.01 * (w ** 2).sum()
gw, gb = T.grad(cost, [w, b])

# compile
learning_rate = 0.1
train = theano.function(
	inputs=[x, y],
	updates=((w, w - learning_rate * gw), (b, b - learning_rate * gb))
)
predict = theano.function(inputs=[x], outputs=prediction)

# # train
# for i in range(500):
# 	train(D[0], D[1])
#
# # save models
# with open('save/model.pickle', 'wb') as f:
# 	model = [w.get_value(), b.get_value()]
# 	pickle.dump(model, f)
# 	print(model[0][:10])
# 	print('accuracy:', compute_accuracy(D[1], predict(D[0])))
	
# load model
with open('save/model.pickle', 'rb') as f:
	model = pickle.load(f)
	w.set_value(model[0])
	b.set_value(model[1])
	print(w.get_value()[:10])
	print('accuracy:', compute_accuracy(D[1], predict(D[0])))
