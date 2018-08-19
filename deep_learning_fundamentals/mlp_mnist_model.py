
from mlp_helper import derivative_W2, derivative_b2, derivative_W1, derivative_b1


def mlp_mnist_model(optimizer, learning_rate=0.001):

	max_iter = 20
	print_period = 10

	X, Y = get_normalized_data()

	learning_rate = 0.00004

	reg = 0.01

	Xtrain = X[:-1000,]
	Ytrain = Y[:-1000]

	Xtest = X[-1000:,]
	Ytest = Y[-1000:]

	# get number of classes
	K = len(set(Y))

	Ytrain_ind = y2indicator(Ytrain, K)
	Ytest_ind = y2indicator(Ytest, K)

	N, D = Xtrain.shape
	batch_size = 500
	n_batches = N / batch_size

	M = 300
	W1 = np.random.randn(D, M)
	b1 = np.zeros(M)
	W2 = np.random.randn(M, K)
	b2 = np.zeros(K)

	catche_W2 = 0
	catche_b2 = 0
	catche_W1 = 0
	catche_b1 = 0

	LL_batch = []
	CR_batch = []

	for itr in range(max_iter):
		count = 0
		for start_b in range(0, N, batch_size):

			count += 1

			end_b = start_b + batch_size

			print("start, end:", start_b, end_b)

			Xbatch = Xtrain[start_b:end_b]
			Ybatch = Ytrain_ind[start_b:end_b]

			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			# pYbatch, Ybatch, Z, 
			# W2 -= learning_rate * (derivative_W2(Ybatch, pYbatch, Z) + reg * W2)
			# b2 -= learning_rate * (derivative_b2(Ybatch, pYbatch) + reg * b2)
			# W1 -= learning_rate * (derivative_W2(Ybatch, pYbatch, W2, Z, Xbatch) + reg * W1)
			# b1 -= learning_rate * (derivative_W2(Ybatch, pYbatch, W2, Z) + reg * b1)

			W1, b1, W2, b2 = GradientDescentOptimizer(Ybatch, pYbatch, Z, W1, b1, W2, b2, reg)

			if count % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)

				ll = cost(pY, Ytest_ind)
				LL_batch.append(ll)
				print("Cost at iteration itr=%d, j=%d: %.6f", (itr, count, ll))

				err = error_rate(pY, Ytest)
				CR_batch.append(err)
				print("Error, rate:", err)

	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print("Final error rate:", error_rate(pY, Ytest))




