
import scipy as sp
import tensorflow as tf

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense



#from sklearn.preprocessing import StandartScaler



class Modelos:

    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        

    def SVM(self):
        clf_svm = SVC(random_state = 42)
        clf_svm.fit(self.X_train.to_numpy(),self.y_train.to_numpy())
        return clf_svm
        
    def SVM_GRID(self):
        param_grid =  {
                'C':[0.5,1,10,100],
                'gamma':['scale',1,0.1,0.01,0.001,0.0001],
                'kernel':['rbf']
            }
        

        svc = SVC()
        clf = GridSearchCV(estimator = svc,param_grid = param_grid,
                          scoring = 'accuracy',n_jobs = 5, cv = 5,verbose = False)
        clf.fit(self.X_train,self.y_train)

        return clf

    def NN(self):
        model = tf.keras.models.Sequential()
        model.add(Dense(10,activation = "relu",name = "layer1"))
        model.add(Dense(5,activation = "relu"))
        model.add(Dense(1,activation = 'softplus'))
        opt = tf.keras.optimizers.SGD(lr = 0.001)
        model.compile(optimizer = opt,loss = 'binary_crossentropy',metrics=  ['accuracy'])
        model.fit(self.X_train.to_numpy().astype(float),self.y_train.to_numpy().astype(int),epochs = 20,verbose= 0)

        return model