import numpy as np
from sklearn.datasets import load_breast_cancer


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(load_breast_cancer().data, load_breast_cancer().target, test_size=0.2,random_state=123)


class LogisticRegression:
    def __init__(self,learning_rate=0.01,iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def sigmoid(self, z):
        '''
        Sigmoid function maps any real value into another
        value between 0 and 1.

                S(z) = 1 / 1 + e** -z
                S(z) = Output between 0 and 1
                z = Input to function (y = mx+b)
                e = base of natural log
        '''
        z = 1 / (1 + np.exp(-z))
        return z

    def cost_function(self,X,y):
        '''
        Mean Absolute Error(L1 Regularization)
        Returns 1D matrix of predictions
        cost_function = sum((y*log(prediction) + (1 - y)*log(1-predictions)) / len(y))
        
        #For label=1
        cost_function = (-y.log(predictions))

        #For label=0
        cost_function = ((1-y)*log(predictions))
        '''
        N = len(X)
        y_predicted = self.sigmoid(np.dot(X,self.coefficients)+self.intercept)
        cost = 1/N * np.sum(-y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        return sum(cost)

    def fit(self,X,y):
        '''
        Gradient Descent 
        dj/dw = 1/N sum(2*x(y - y_predicted))
        dj/db = 1/N sum(2(y - y_predicted))
        
        To update weights
        w = w - learning_rate * dw
        b = b - learning_rate * db
        '''
        loss = []
        N = len(X)
        self.coefficients = np.ones(shape=(X.shape[1]))
        self.intercept = 0

        
        for i in range(1,self.iterations+1):
            y_predicted = self.sigmoid(np.dot(X,self.coefficients)+self.intercept)
            
            dw = (1/N) * np.dot(X.T,(y_predicted - y))
            db = (1/N) * np.sum(y_predicted - y)
            
            self.coefficients -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db
            
        
    def predict(self,X_test):
        y_predicted = self.sigmoid(np.dot(X_test,self.coefficients)+self.intercept)
        y_predicted_class =  [1 if i>=0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def accuracy(self,y_test,y_predicted):
        accuracy = np.sum(y_test == y_predicted) / len(y_test)
        return accuracy

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)
print("Accuracy of LogisticRegression Model: ",model.accuracy(y_test,y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))