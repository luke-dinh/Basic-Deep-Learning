import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sig_derivative(x):
    return x*(1-x)
class NeuralNetwork:
    def init(self,layers,alpha = 0.05):
        self.layers = layers
        #Learning rate
        self.alpha = alpha
        # set up w and b
        self.W = []
        self.b = []
    #set up bias in NN
    for i in range(0,(len(layers)-1)):
        w = np.random.rand(layers[i],layers[i+1])
        b = np.zeros((layers[i+1],1))
        self.W.append(w/layers[i])
        self.b.append(b)
    #summarize the NN
    def repr(self):
        return "NeuralNetwork{[]}".format("-".join(str(l)) for l in self.layers())
    def fit_partial(self,x,y):
        #A==L
        A = [x]
        out = A[-1]
        for i in range (0, (len(self.layers)-1)):
            out = sigmoid(np.dot(out,self.W[i])) + ((self.b[i])).T
            A.append(out)
        y = y.reshape(-1,1)
        dA = ((-y/A[-1]) - (1-y)/(1-A[-1]))
        dW = []
        db = []
        for i in reversed(0, range(len(self.layers)) -1):
            dw_ = np.dot((A[i]).T, dA[-1] * sig_derivative(A[i+1]))
            db_ =(np.sum(dA[-1]*sig_derivative(A[i+1]),0))
            dA_ = np.dot(((sig_derivative(A[i+1])* dA[-1])), (self.W[i]).T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)
        dW = dW[::-1]
        db = db[::-1]
        # Gradient descent
        for i in range(0, len(self.layers)-1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]
        def fit(self, X, y, epochs=20, verbose=10):
            for epoch in range(0, epochs):
                self.fit_partial(X, y)
                if epoch % verbose == 0:
                    loss = self.calculate_loss(X, y)
                    print("Epoch {}, loss {}".format(epoch, loss))
        #Prediction
        def predict(self, X):
            for i in range(0, len(self.layers) - 1):
                X = sigmoid(np.dot(X, self.W[i]) + (self.b[i].T))
                return X
        #Loss Function
        def loss(self,X,y):
            y_predict = self.predict(X)
            return -(np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict)))

