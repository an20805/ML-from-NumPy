import numpy as np

def unit_step_fn(x):
    return np.where(x>0, 1, 0)

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.activation_fn = unit_step_fn

    # data should be linearly separable

    def fit(self, X, y):
        n_samples , n_features = X.shape

        self.weights = np.random.normal(size=n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            for ind, x in enumerate(X):
                linear_output = np.dot(x, self.weights) + self.bias
                
                y_predicted = self.activation_fn(linear_output)

                step_size = self.lr * (y[ind] - y_predicted)
                self.weights += step_size * x
                self.bias += step_size
        

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_fn(linear_output)
        return y_predicted
    
# For testing

if __name__ == "__main__":
    
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn.metrics import classification_report

    X, y = datasets.make_blobs(n_samples=200, n_features=5, centers=2, random_state=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = Perceptron()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print('Feature Weights: ', clf.weights)
    print('Bias: ', clf.bias)


