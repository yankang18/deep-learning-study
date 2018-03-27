
import numpy as np

def GradientDescentOptimizer(Ybatch, pYbatch, Z, W1, b1, W2, b2, reg=0.01):

	# pYbatch, Ybatch, Z, 
	W2 -= learning_rate * (derivative_W2(Ybatch, pYbatch, Z) + reg * W2)
	b2 -= learning_rate * (derivative_b2(Ybatch, pYbatch) + reg * b2)
	W1 -= learning_rate * (derivative_W1(Ybatch, pYbatch, W1, Z, Xbatch) + reg * W1)
	b1 -= learning_rate * (derivative_b1(Ybatch, pYbatch, b1, Z) + reg * b1)

	return W1, b1, W2, b2


def RMSPropOptimizer(Xbatch, Ybatch, pYbatch, Z, W1, b1, W2, b2, 
                     cache_W1, cache_b1, cache_W2, cache_b2, reg=0.01, learning_rate=0.001):

    eps = 0.000001
    decay_rate = 0.999
    
    # calculate gradient descent
    gW2 = derivative_W2(Ybatch, pYbatch, Z) + reg * W2
    # update cache
    cache_W2 = decay_rate * cache_W2 + (1 - decay_rate) * gW2 * gW2
    # udpate weight
    W2 -= learning_rate * gW2 / (np.sqrt(cache_W2) + eps)

    gb2 = derivative_b2(Ybatch, pYbatch) + reg * b2
    cache_b2 = decay_rate * cache_b2 + (1 - decay_rate) * gb2 * gb2
    b2 -= learning_rate * gb2 / (np.sqrt(cache_b2) + eps)

    gW1 = derivative_W1(Ybatch, pYbatch, W2, Z, Xbatch) + reg * W1
    cache_W1 = decay_rate * cache_W1 + (1 - decay_rate) * gW1 * gW1
    W1 -= learning_rate * gW1 / (np.sqrt(cache_W1) + eps)

    gb1 = derivative_b1(Ybatch, pYbatch, W2, Z) + reg * b1
    cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1 * gb1
    b1 -= learning_rate * gb1 / (np.sqrt(cache_b1) + eps)

    return W1, b1, W2, b2, cache_W1, cache_b1, cache_W2, cache_b2


def AdamOptimizer(Ybatch, pYbatch, Z, W1, b1, W2, b2, 
	mW1, vW1, mb1, vb1, mW2, vW2, mb2, vb2, reg=0.001, learning_rate=0.001, t):


	eps = 1e-8
	beta_m = 0.9
	beta_v = 0.999


	# gradients
	gW2 = derivative_W2(Ybatch, pYbatch, Z) + reg * W2
	gb2 = derivative_b2(Ybatch, pYbatch) + reg * b2
	gW1 = derivative_W1(Ybatch, pYbatch, W2, Z, Xbatch) + reg * W1
	gb1 = derivative_b1(Ybatch, pYbatch, W2, Z) + reg * b1

	# new m
	mW2 = beta_m * mW2 + (1 - beta_m) * gW2
	mb2 = beta_m * mb2 + (1 - beta_m) * mb2
	mW1 = beta_m * mW1 + (1 - beta_m) * gW1
	mb1 = beta_m * mb1 + (1 - beta_m) * gb1

	# new v
	vW2 = beta_v * vW2 + (1 - beta_v) * gW2 * gW2
	vb2 = beta_v * vb2 + (1 - beta_v) * vb2 * vb2
	vW1 = beta_v * vW1 + (1 - beta_v) * gW1 * gW1
	vb1 = beta_v * vb1 + (1 - beta_v) * gb1 * gb1


	# correction
	correction_m = 1 - beta_m ** t
	corr_mW2 = mW2 / correction_m
	corr_mb2 = mb2 / correction_m
	corr_mW1 = mW1 / correction_m
	corr_mb1 = mb1 / correction_m

	correction_v = 1 - beta_v ** t
	corr_vW2 = vW2 / correction_v
	corr_vb2 = vb2 / correction_v
	corr_vW1 = vW1 / correction_v
	corr_vb1 = vb1 / correction_v

	# apply updates to the params
	W2 -= learning_rate * corr_mW2 / np.sqrt(corr_vW2 + eps)
	b2 -= learning_rate * corr_mb2 / np.sqrt(corr_vb2 + eps)
	W1 -= learning_rate * corr_mW1 / np.sqrt(corr_vW1 + eps)
	b1 -= learning_rate * corr_mb1 / np.sqrt(corr_vb1 + eps)

	return W1, b1, W2, b2, mW1, vW1, mb1, vb1, mW2, vW2, mb2, vb2


