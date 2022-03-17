
import scipy as sp
import tensorflow as tf

from sklearn.svm import SVC
from tensorflow.keras.layers import Dense


#from sklearn.preprocessing import StandartScaler



class Modelos:

    def __init__(self) -> None:
        pass

    def SVM(self,X_train,y_train):
        clf_svm = SVC(random_state = 42)
        clf_svm.fit(X_train.to_numpy(),y_train.to_numpy())
        return clf_svm
        
    def NN(self,X_train,y_train):
        model = tf.keras.models.Sequential()
        model.add(Dense(10,activation = "relu",name = "layer1"))
        model.add(Dense(5,activation = "relu"))
        model.add(Dense(1,activation = 'softplus'))
        opt = tf.keras.optimizers.SGD(lr = 0.001)
        model.compile(optimizer = opt,loss = 'binary_crossentropy',metrics=  ['accuracy'])
        model.fit(X_train.to_numpy().astype(float),y_train.to_numpy().astype(int),epochs = 20,verbose= 0)

        return model