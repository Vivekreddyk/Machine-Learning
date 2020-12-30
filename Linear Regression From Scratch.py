import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(load_boston().data, load_boston().target, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)


class LinearRegression:
    """
    Parameters
    -------------
    learning_rate: float
        The step length that will be used when updating weights.

    n_iterations: float
        The number of training iterations the algorithm will tune the weight for.

    """
    def __init__(self,learning_rate=0.01,iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def predict(self,X_test):
        y_predicted = np.dot(X_test,self.weights) + self.bias #y = mx + b (Line Equation)
        return y_predicted
    
    def fit(self,X,y):
        self.weights = np.ones(shape=(X.shape[1]))  #Coefficients
        self.bias = np.zeros(shape=(1))             #Intercept
        N = len(X)                                  #Total Values
        
        for _ in range(1,self.iterations+1):        #Iterating over each epoch
            for i in range(N):                      #Iterating over each value
                y_predicted = np.dot(X,self.weights) + self.bias
                
                dw = -(2/N) * np.dot(X.T,(y - y_predicted))  
                db = -(2/N) * np.sum((y - y_predicted))
                
                self.weights = self.weights - self.learning_rate * dw   
                self.bias = self.bias - self.learning_rate * db
                
                loss = np.mean(np.square(y - y_predicted))
                
            print("Epoch:{}  Loss:{}".format(_,loss))

model = LinearRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)  
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)