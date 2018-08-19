

# softmax(sigmoid(X*W1 + b1)*W2 + b2)
def forward(X, W1, b1, W2, b2):
    Z = sigmoid(X.dot(W1) + b1)
    Y = softmax(Z.dot(W2) + b2)  
    return Y, Z


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)



def derivative_W2(T, Y, Z):
    return Z.T.dot(Y - T)



def derivative_b2(T, Y):
    return (Y - T).sum(axis=0)
    
    

def derivative_W1(T, Y, W2, Z, X):
    dA = (Y - T).dot(W2.T) * Z * (1 - Z)
    return X.T.dot(dA)


def derivative_b1(T, Y, W2, Z):
    dz = (Y - T).dot(W2.T) * Z * (1 - Z)
    return dz.sum(axis=0)

def y2indicator(y, k):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, k))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)


def cross_entropy(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()