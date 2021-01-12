import numpy as np
from sklearn.datasets import load_iris,load_boston
from sklearn.model_selection import  train_test_split

X_train,X_test,y_train,y_test = train_test_split(load_iris().data,load_iris().target,test_size=0.2,random_state=42)

class KNeighborsClassifier:
    """ 
    KNeighborsClassifier
    Parameters:
    -----------
    K: int
        The number of closest neighbors that will determine the class of the 
         each sample that we try to predict

    """

    def __init__(self,K=3):
        self.K = K

    def __euclidean_distance(self,X_test):
        """
        Calculating the Euclidean distance
        Row --> Test Example
        Column --> Distance of that example
        Shape = array(30,120)

        """
        distances = np.empty((X_test.shape[0],self.X_train.shape[0]))
        
        for i in range(X_test.shape[0]):
            for j in  range(self.X_train.shape[0]):
                distances[i,j] = np.sqrt(np.sum((X_test[i,:] - self.X_train[j,:])**2))
        return distances

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def __predict_labels(self,X_test):
        y_pred = np.zeros(X_test.shape[0]) #Shape(30,4)
        
        #determine each sample
        for i in range(X_test.shape[0]):
            #Sort the test samples
            indices = np.argsort(X_test[i,:]) 
            #Sort the training samples by their distance to test samples and get the K nearest neighbor 
            closest_labels = self.y_train[indices[:self.K]]
            #Get the most common class label
            y_pred[i] = np.argmax(np.bincount(closest_labels))
        return y_pred.astype(int)
    
    def predict(self,X_test):
        distances = self.__euclidean_distance(X_test)
        return self.__predict_labels(distances)


model = KNeighborsClassifier()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(predictions)
accuracy = np.sum(predictions==y_test)/len(y_test)
print(accuracy*100) 