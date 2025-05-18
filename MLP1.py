# -*- coding: utf-8 -*-
"""
Created on Mon May 12 12:05:57 2025

@author: Potito
"""

import numpy as np

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# %% loading digits dataset
"""

data : 2D array of ints, >=0 and <=16
    for each sample (rows), contains the corresponding features (columns).
target : 1D array of ints, >=0 and <10
    for each sample (indeces), contains the corresponding label.
    
"""
data, target = load_digits(return_X_y=True)
# %%

# %% splitting dataset into training and test set
"""

X_train : 2D array of ints, >=0 and <=16
    for each sample (rows) from training set, contains the corresponding 
    features (columns).
X_test : 2D array of ints, >=0 and <=16
    for each sample (rows) from test set, contains the corresponding features 
    (columns).
y_train : 1D array of ints, >=0 and <10
    for each sample (indeces) in the training set label, contains the 
    corresponding label.
y_test : 1D array of ints, >=0 and <10
    for each sample (indeces) in the test set label, contains the corresponding
    label.
    
"""
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
# %%

# %% normalizing inputs from training and test set
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%

# %% MLP Classifier class
class MLPClassifier():
        
    def __init__(self, hidden_layers=(100,), activation='relu', eta=0.001):
        """
        Constructs a multi-layer perceptron classifier object.

        Parameters
        ----------
        hidden_layers : tuple of ints, optional
            contains for the input layer, hidden layers and output layer, 
            respectively the number of features and neurons. The default is 
            (100,).
        activations : str, optional
            a string which indicates the activation function for each hidden 
            layer. The default is 'relu'.
        eta : float, optional
            learning rate of net. The default is 0.001

        Returns
        -------
        None.

        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.eta = eta
        
        self.layers = []
        self.loss_fn = Loss('x_entropy')
        
    def initialize_net(self, input_dim, output_dim):
        """
        Initialize the hidden layers and output layer of the net.

        Parameters
        ----------
        input_dim : int
            number of input features.
        output_dim : int
            number of output classes.

        Returns
        -------
        None.

        """
        self.layers.clear()
        self.n_class = output_dim
        for i in self.hidden_layers:
            self.add(input_dim, i, self.activation)
            input_dim = i
        self.add(input_dim, output_dim, out=True)
        
    @staticmethod
    def count_unique(lst):
        """
        Count the number of unique element in the input list.

        Parameters
        ----------
        lst : list of elements
            list of elements.

        Returns
        -------
        int
            number of unique elements in lst.

        """
        elems = []
        for e in lst:
            if e not in elems:
                elems.append(e)
        return len(elems)
        
    def add(self, input_dim, n_units, activation='softmax', out=False):
        """
        Add a layer to this MLPClassifier object.

        Parameters
        ----------
        input_dim : int >0
            input dimension.
        n_units : int >0
            number of neurons of the layer.
        activation : str, optional
            indicates activation function of the layer. The default is None.

        Returns
        -------
        None.

        """
        self.layers.append(Layer(input_dim, n_units, activation, out))
        
    def predict_proba(self, X):
        """

        Parameters
        ----------
        X : 2D array
            for each example (rows), contains the corresponding features 
            (columns).

        Returns
        -------
        proba : 2D array
            for each example (rows), contains the probabilities (columns) of 
            belonging to a class.

        """
        proba = X
        for l in self.layers:
            proba = l.forward(proba)
        return proba
    
    def predict(self, X):
        """
        

        Parameters
        ----------
        X : 2D array
            for each example (rows), contains the corresponding features 
            (columns).

        Returns
        -------
        pred : array of ints
            for each sample (indices) in X, contains the predicted class to 
            which it belongs.

        """
        proba = self.predict_proba(X)
        pred = np.argmax(proba, axis=1)
        return pred
    
    def fit(self, X, y, epochs):
        """
        
        
        Parameters
        ----------
        X : 2D array
            for each sample (rows), contains the corresponding features 
            (columns).
        y : array of positive integers, less than the number of classes
            for each sample (length of array), contains the corresponding 
            label.
        epochs : positive int
            number of epochs for which training the net.

        Returns
        -------
        None.

        """
        n_features, n_classes = X.shape[1], MLPClassifier.count_unique(y)
        self.initialize_net(n_features, n_classes)
        
        y_one_hot = np.eye(n_classes)[y]
        
        for i in range(epochs):
            self.predict_proba(X)
            for l in self.layers[::-1]:
                l.backward(l.input, y_one_hot, self.eta)
            Layer.prev_weights = None
    
    def score(self, X, y):
        """
        

        Parameters
        ----------
        X : 2D array
            for each sample (rows), contains the corresponding features 
            (columns).
        y : array of positive integers, less than the number of classes
            for each sample (length of array), contains the corresponding 
            label.

        Returns
        -------
        accuracy : float
            the number of correct classified samples in X, over the samples 
            number.

        """
        pred = self.predict(X)
        accuracy = np.mean(pred == y)
        return accuracy
    
# %%

# %% Layer class
class Layer():
    
    # implementalo come pila o coda
    prev_weights = None
    
    def __init__(self, input_dim, n_units, activation='linear', out=False):
        """
        constructs a Layer object

        Parameters
        ----------
        input_dim : int >0
            input dimension.
        n_units : int >0
            number of neurons of the layer.
        activation : str, optional
            indicates activation function of the layer. The default is 
            'linear'.
        out : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        # np.random.seed(42)
        self.input_dim = input_dim
        self.units = n_units
        # self.w = np.random.rand(n_units, input_dim)
        # self.b = np.random.rand(n_units)
        limit = np.sqrt(6 / (input_dim + n_units))
        self.w = np.random.uniform(-limit, limit, (n_units, input_dim))
        self.b = np.zeros(n_units)
        self.activation = Activation(activation)
        self.gradient = Gradient(activation, out)
        
    def forward(self, inputs):
        """
        

        Parameters
        ----------
        inputs : 2D array of floats
            for each sample (rows), contains the features/activations (columns)
            from the input/previous layer.

        Returns
        -------
        output : 2D array of floats
            for each sample (rows), contains the activations (columns) 
            calculated in this layer.

        """
        self.input = inputs
        z = np.dot(inputs, self.w.T) + self.b
        self.output = self.activation(z)
        return self.output
    
    def backward(self, X, y, eta):
        
        nabla_w, nabla_b = self.gradient(X, y, self.output, Layer.prev_weights)
        Layer.prev_weights = self.w.copy()
        
        dw = eta * nabla_w
        db = eta * nabla_b
        
        self.w -= dw
        self.b -= db
        
# %%

# %% Activation
class Activation():
    
    def __init__(self, activation):
        """
        constructs a Activation object

        Parameters
        ----------
        activation : str
            indicates activation function of this object.

        Returns
        -------
        None.

        """
        self.name = activation
        if activation == 'relu':
            self.fn = Activation.relu
        elif activation == 'softmax':
            self.fn = Activation.softmax
        elif activation == 'sigmoid':
            self.fn = Activation.sigmoid
        else:
            self.fn = lambda x: x
            
    def __call__(self, z):
        return self.fn(z)
    
    @staticmethod
    def relu(z):
        return np.maximum(0,z)
    
    @staticmethod
    def softmax(z):
        """
        
        Parameters
        ----------
        z : 2D array of floats
            for each sample (rows), contains the non-normalized probabilities 
            of belonging to a class (columns).

        Returns
        -------
        2D array of floats, >0 and <1
            for each sample (rows), contains the probabilities (columns) of 
            belonging to a class.

        """
        z = z - np.max(z, axis=1, keepdims=True)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
# %%

# %% Loss function class
class Loss():
    
    def __init__(self, loss_fn):
        """
        constructs a Loss object

        Parameters
        ----------
        loss_fn : str
            indicates the loss function to initialize this object.

        Returns
        -------
        None.

        """
        self.name = loss_fn
        if loss_fn == 'x_entropy':
            self.fn = Loss.cross_entropy
            
    def __call__(self, y, pred):
        return self.fn(y, pred)
    
    @staticmethod
    def cross_entropy(y, pred):
        epsilon = 1e-10
        return -(np.sum(y * np.log(pred + epsilon)) / y.shape[0])
# %%

# %%
class Gradient():
    
    # n_sample * n_neuron
    part_grad = None
    
    def __init__(self, activation, out=False):
        if out:
            if activation == 'softmax':
                self.fn = Gradient.out_softmax
        else:
            if activation == 'sigmoid':
                self.fn = Gradient.hidden_sigmoid
            elif activation == 'relu':
                self.fn = Gradient.hidden_relu
                
    def __call__(self, X, y, output, w=None):
        return self.fn(X, y, output, w)
        
    @staticmethod
    def out_softmax(X, y, proba, w):
        
        Gradient.part_grad = proba - y
        
        # la y matrice di vettori one-hot
        nabla_w = np.dot((proba - y).T, X) / X.shape[0]
        nabla_b = np.sum(proba - y, axis=0) / y.shape[0]
        
        return nabla_w, nabla_b
    
    @staticmethod
    def hidden_sigmoid(X, y, output, w):
        
        prev_part_grad = Gradient.part_grad
        Gradient.part_grad = np.dot(prev_part_grad, w) * (output * (1 - output))
        
        nabla_w = np.dot(Gradient.part_grad.T, X) / X.shape[0]
        nabla_b = np.sum(Gradient.part_grad, axis=0) / output.shape[0]
        
        return nabla_w, nabla_b
    
    @staticmethod
    def hidden_relu(X, y, output, w):
        
        prev_part_grad = Gradient.part_grad
        
        # derivata degli output della funzione ReLU
        out_cpy = output.copy()
        out_cpy[out_cpy > 0] = 1
        out_cpy[out_cpy <= 0] = 0
        
        Gradient.part_grad = np.dot(prev_part_grad, w) * out_cpy
        
        nabla_w = np.dot(Gradient.part_grad.T, X) / X.shape[0]
        nabla_b = np.sum(Gradient.part_grad, axis=0) / output.shape[0]
        
        return nabla_w, nabla_b
        
# %%

mlp = MLPClassifier(hidden_layers=(128, 256, 512), eta=0.04)
mlp.fit(X_train, y_train, 2000)
accuracy = mlp.score(X_test, y_test)
print(f"accuracy : {accuracy * 100}%")
print(f'accuracy on train set : {mlp.score(X_train, y_train) * 100}%')