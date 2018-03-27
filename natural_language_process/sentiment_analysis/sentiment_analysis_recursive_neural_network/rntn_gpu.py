import sys
import theano
import theano.tensor as T
import numpy as np
import string
import json
import operator
from sklearn.utils import shuffle
from datetime import datetime


def main(train, test, word2idx, dim=10, is_binary=True, learning_rate=1e-2, reg=1e-2, 
         mu=0, eps=1e-2, activation=T.tanh, epochs=30, train_inner_nodes=False):
    
    print("total train size before filtering:", len(train))
        
    if is_binary:
        train = [t for t in train if t[3][-1] >=0]
        test = [t for t in test if t[3][-1] >=0]
    
    print("total train size after filtering:", len(train))
    
    train = shuffle(train)
#     train = train[:5000]
    
    test = shuffle(test)
#     test = test[:1000]
    
    print("train size:", len(train))
    print("test size:", len(test))   
    
    V = len(word2idx)
    print("vocab size:", V)
    D = dim
    K = 2 if is_binary else 5
    
    model = RecursiveNN(V, D, K)
    model.fit(train, learning_rate=learning_rate, reg=reg, mu=mu, eps=eps, epochs=epochs, activation=activation, train_inner_nodes=train_inner_nodes)
    print ("train accuracy:", model.score(train))
    print ("test accuracy:", model.score(test))




class RecursiveNN(object):
    def __init__(self, V, D, K):
        self.V = V
        self.D = D
        self.K = K
        
    def fit(self, trees, learning_rate=1e-3, reg=1e-2, mu=0.99, eps=1e-2, decay_rate=0.999, epochs=20, 
            activation=T.nnet.relu, train_inner_nodes=True):
        

        learning_rate = np.float32(learning_rate)
        reg = np.float32(reg)
        mu = np.float32(mu)
        eps = np.float32(eps)
        decay_rate = np.float32(decay_rate)

        print("learning rate:", learning_rate)
        print("regularization:", reg)
        print("mu:", mu)
        print("eps:", eps)
        print("epochs:", epochs)
        print("decay_rate:", decay_rate)
        print("dim:", self.D)
        print("activation: ", type(activation))
        print("train_inner_nodes: ", train_inner_nodes)
        
        V = self.V
        D = self.D
        K = self.K
        R = 2
        self.f = activation
        N = len(trees)
        
        ### initialize weights
        
        We = init_weights(V, D)
        
        Wh = np.random.randn(R, D, D) / np.sqrt(R + D + D)
        bh = np.zeros(D)
        
        Wo = init_weights(D, K)
        bo = np.zeros(K)
        
        self.We = theano.shared(We.astype('float32'), 'We')
        self.Wh = theano.shared(Wh.astype('float32'), 'Wh')
        self.bh = theano.shared(bh.astype('float32'), 'bh')
        self.Wo = theano.shared(Wo.astype('float32'), 'Wo')
        self.bo = theano.shared(bo.astype('float32'), 'bo')
        self.params = [self.We, self.Wh, self.bh, self.Wo, self.bo]
        
        
        ### symbolic expression for forward propagation
            
        # create input training vectors
#         words = T.ivector('words')
#         parents = T.ivector('parents')
#         relations = T.ivector('relations')
#         labels = T.ivector('labels')

        words = T.ivector('words')
        parents = T.ivector('parents')
        relations = T.ivector('relations')
        labels = T.ivector('labels')
        
        def recurrence(n, hiddens, words, parents, relations):
          
            w = words[n]
        
            # update hidden matrix for current node
            hiddens = T.switch(
                T.ge(w, 0),
                T.set_subtensor(hiddens[n], self.We[w]),
                T.set_subtensor(hiddens[n], self.f(hiddens[n] + self.bh)),
            )
            
            # update hidden matrix for parent node
            p = parents[n]
            r = relations[n]
            hiddens = T.switch(
                T.ge(p, 0),
                T.set_subtensor(hiddens[p], hiddens[p] + hiddens[n].dot(self.Wh[r])),
                hiddens,
            )
            
            return hiddens
        
        # initialize hidden matrix
        # note that each row of the hidden matrix represents a node in the original parse tree
        # it can be leave node containing word or inner node
        hiddens = T.zeros((words.shape[0], D), dtype='float32')
        
        h, _ = theano.scan(
            fn=recurrence,
            sequences=T.arange(words.shape[0]),
            n_steps=words.shape[0],
            outputs_info=[hiddens],
            non_sequences=[words, parents, relations],
        )
        
        # note we use T.arange not python's range and use T.zeros not np.zeros below
        
        # symbolic expression of the output probablility distribution
        py_x = T.nnet.softmax(h[-1].dot(self.Wo) + self.bo)
        prediction = T.argmax(py_x, axis=1)
        
        ### symbolic expression for back propagation 
        
        # regularization cost
        rcost = reg*T.mean([(p*p).sum() for p in self.params])
        
        if train_inner_nodes:
            # won't work for binary classification
            xentropy = -T.mean(T.log(py_x[T.arange(labels.shape[0]), labels]))
            acost = xentropy + rcost
        else:
            xentropy = -T.mean(T.log(py_x[-1, labels[-1]]))
            acost = xentropy + rcost
        
        grads = T.grad(acost, self.params)
        
        # momentum
#         dparams = [theano.shared(p.get_value()*0) for p in self.params]
#         updates = [
#             (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
#         ] + [
#             (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
#         ]
        
        # AdaGrad
        cache = [theano.shared(p.get_value()*0) for p in self.params]
        updates = [
            (c, c + g*g) for c, g in zip(cache, grads)
        ] + [
            (p, p - learning_rate*g / (T.sqrt(c) + eps)) for p, c, g in zip(self.params, cache, grads)
        ]
        
        # RMSprop (does not work! why???)
#         cache = [theano.shared(p.get_value()*0) for p in self.params]
#         updates = [
#             (c, decay_rate * c + (1 - decay_rate)*g*g) for c, g in zip(cache, grads)
#         ] + [
#             (p, p - learning_rate*g / (T.sqrt(c) + eps)) for p, c, g in zip(self.params, cache, grads)
#         ]


        # RMSprop + momentum
#         dparams = [theano.shared(p.get_value()*0) for p in self.params]
#         cache = [theano.shared(p.get_value()*0) for p in self.params]
#         updates = [
#             (c, decay_rate * c + (1 - decay_rate)*g*g) for c, g in zip(cache, grads)
#         ] + [
#             (dp, mu*dp - learning_rate*g / (T.sqrt(c) + eps)) for dp, c, g in zip(dparams, cache, grads)
#         ] + [
#             (p, p + dp) for p, dp in zip(self.params, dparams)
#         ]
        
        
        self.cost_predict_op = theano.function(
            inputs = [words, parents, relations, labels],
            outputs = [acost, prediction],
            allow_input_downcast=True,
        )
        
        self.train_op = theano.function(
            inputs = [words, parents, relations, labels],
            outputs = [xentropy, acost, prediction],
            updates=updates,
            allow_input_downcast=True,
        ) 
        
        ### start training
        print("start training...")
        
        costs = []
        xents = []
        sequence_indexes = range(N)
        if train_inner_nodes:
            n_total = sum(len(words) for words, _, _, _ in trees)
        else:
            n_total = N
            
        for i in range(epochs):
            t0 = datetime.now()
            sequence_indexes = shuffle(sequence_indexes)
            n_correct = 0
            cost = 0
            xent = 0
            it = 0 # iteration count
            for j in sequence_indexes:
                words, parents, relations, labels = trees[j]
                xe, c, p = self.train_op(words, parents, relations, labels)
                
                if np.isnan(c):
                    print("Cost is nan! Let's stop here. Why don't you try decreasing the learning rate?")
                    exit()
                    
                cost += c
                xent += xe
                if train_inner_nodes:
                    n_correct += np.sum(p == labels)
                else:
                    n_correct += (p[-1] == labels[-1])
                it+=1
                if it % 5 == 0:
                    sys.stdout.write("epoch: %d, j/N: %d/%d correct rate so far: %f, cost so far: %f, xent so far: %f\r" % (i, it, N, float(n_correct)/n_total, cost, xent))
                    sys.stdout.flush()
            
            print("i:", i, "cost:", cost, "xent", xent, "correct rate:", (float(n_correct)/n_total), "time for epoch:", (datetime.now() - t0))
            costs.append(cost)
            xents.append(xent)
            
#         plt.plot(costs)
#         plt.title("cost")
#         plt.show()
        
#         plt.plot(xents)
#         plt.title("cross entropy")
#         plt.show()
        
    def score(self, trees):
        n_total = len(trees)
        n_correct = 0
        for words, parents, relations, labels in trees:
            _, p = self.cost_predict_op(words, parents, relations, labels)
            n_correct += (p[-1] == labels[-1])
        print("n_correct:", n_correct, "n_total:", n_total)
        return float(n_correct) / n_total
    
    def f1_score(self, trees):
        Y = []
        P = []
        for words, left, right, lab in trees:
            _, p = self.cost_predict_op(words, left, right, lab)
            Y.append(lab[-1])
            P.append(p[-1])
        return f1_score(Y, P, average=None).mean()
    



def init_weights(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


def load_data(data_file=None):
    if data_file == None:
        return
    with open(data_file) as f:
        data = json.load(f)
    return data


if __name__ == '__main__':

	# folder = './data/large_files/stanford_sentiment/parsed_data/'
	folder = './parsed_data/'
	word2idx = load_data(folder + "sentiment_word2idx.json")
	sentiment_binary_train = load_data(folder + "sentiment_binary_train.json")
	sentiment_train = load_data(folder + "sentiment_train.json")
	sentiment_binary_test = load_data(folder + "sentiment_binary_test.json")
	sentiment_test = load_data(folder + "sentiment_test.json")

	print(len(sentiment_binary_train))
	print(len(sentiment_binary_test))
	print("Load data finished")	

	train = list(sentiment_binary_train.values()) 
	test = list(sentiment_binary_test.values()) 
	print(len(train))
	print(len(test))
	print("preprocess data finished")

	main(train, test, word2idx, dim=10, learning_rate=1e-2, reg=1e-2, mu=0, eps=1e-2, activation=T.tanh, epochs=40, train_inner_nodes=False)
