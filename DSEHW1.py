import numpy as np

class DSELinearClassifier():
    
    """Linear classifiers.
    
    Parameters
    ------------
    activation: string
      The type of activation method (Perceptron, Logistic , or HyperTan)
      Required.
    learning_rate : float
      Learning rate (between 0.0 and 1.0)
    random_state : int
      Random number generator seed for random weight initialization.
    
    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    fit_errors_ : list
      Number of misclassifications (updates) in each epoch.
    
    """
        
    def __init__(self, activation, random_state=42, initial_weight=None, learning_rate=4):
        self.activation= activation
        self.initial_weight=initial_weight
        self.learning_rate = learning_rate
        self.random_state = random_state
        
    def fit(self, X, y, max_epochs=20, batch_size = None):
        """ Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples
            is the number of examples and
            n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.
        
        Returns
        -------
        self : object
        
        """
        self.fit_errors_ = []

        rgen = np.random.RandomState(self.random_state)
        if self.initial_weight == None:
            self.w_ = rgen.normal(loc=0.0, scale=0.01,
                                  size=1 + X.shape[1])
        else:
            self.w_ = self.initial_weight
        
        if self.activation == "Perceptron": 
                for _ in range(max_epochs):
                    errors = 0
                    for xi, target in zip(X, y):
                        update = self.learning_rate * (target - self.predict(xi))
                        self.w_[1:] += update * xi
                        self.w_[0] += update
                        errors += int(update != 0.0)
                    self.fit_errors_.append(errors)
                return self
        
        elif self.activation == 'Logistic':
                for i in range(max_epochs):
                    net_input = self.net_input(X)
                    output = self.activation_function(net_input)
                    errors = (y - output)
                    self.w_[1:] += self.learning_rate * X.T.dot(errors)
                    self.w_[0] += self.learning_rate * errors.sum()

                    # note that we compute the logistic `cost` now
                    # instead of the sum of squared errors cost
                    cost = (-y.dot(np.log(output)) -
                                ((1 - y).dot(np.log(1 - output))))
                    self.fit_errors_.append(cost)
                return self
        
        elif self.activation == 'HyperTan':
                for i in range(max_epochs):
                    net_input = self.net_input(X)
                    output = self.activation_function(net_input)
                    errors = (y - output)
                    self.w_[1:] += self.learning_rate * X.T.dot(errors)
                    self.w_[0] += self.learning_rate * errors.sum()

                    # note that we compute the logistic `cost` now
                    # instead of the sum of squared errors cost
                    cost = (-y.dot(np.log(output)) -
                                ((1 - y).dot(np.log(1 - output))))
                    self.fit_errors_.append(cost)
                return self
        
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation_function(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        """Return class label after unit step"""
        if self.activation == 'Perceptron':
            return np.where(self.net_input(X) >= 0.0, 1, -1)
        elif self.activation == 'Logistic':
            return np.where(self.net_input(X) >= 0.0, 1, 0)
        elif self.activation == 'HyperTan':
            return np.where(self.net_input(X) >= 0.0, 1, -1)
            