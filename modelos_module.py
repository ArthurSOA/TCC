
import scipy as sp
import tensorflow as tf
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout






#from sklearn.preprocessing import StandartScaler



class Modelos:

    def __init__(self,X_train,y_train, X_val = None,y_val = None):
        if isinstance(X_train, pd.DataFrame):
            self.X_train = X_train.to_numpy()
            self.y_train = y_train.to_numpy()
            if X_val is not None:
                self.X_val = X_val.to_numpy()
                self.y_val = y_val.to_numpy()
        else:
            self.X_train = X_train
            self.y_train = y_train.to_numpy()
            if X_val is not None:
                self.X_val = X_val
                self.y_val = y_val.to_numpy()


       
        

    def SVM(self):
        clf_svm = SVC(random_state = 42,kernel='rbf')
        clf_svm.fit(X=self.X_train,y=self.y_train)
        return clf_svm
        
    def SVM_GRID(self):
        param_grid =  {
                'C':[0.1,0.3,0.5,1,10,100],
                'gamma':['scale',1,0.1,0.01,0.001],
                'kernel':['rbf']
            }
        

        svc = SVC()
        clf = GridSearchCV(estimator = svc,param_grid = param_grid,
                          scoring = 'accuracy',n_jobs = 5, cv = 5,verbose = False)
        clf.fit(self.X_train,self.y_train)

        return clf


    def grid_nn(self,neuronios: list = [10,20,30,40,50,80]):  
        modelos = []
        for i in neuronios:
            modelos.append(self.NN(i))

        return modelos


    def NN(self,camada2):


        model = tf.keras.models.Sequential()
        model.add(Dense(20,activation = 'relu',input_dim = self.X_train.shape[1]))
        model.add(Dense(camada2,activation = 'relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(40,activation = "relu"))
        #model.add(Dropout(0.5))
        #model.add(Dense(20,activation = "relu"))
        #model.add(Dropout(0.5))
        #model.add(Dense(30,activation = 'relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(1,activation = 'sigmoid'))
        #opt = tf.keras.optimizers.SGD(lr = 0.001)
        opt = tf.keras.optimizers.Adam()
        model.compile(optimizer = opt,loss = 'binary_crossentropy',metrics=  ['accuracy'])
        model.fit(self.X_train.astype(float),self.y_train.astype(int),epochs = 300,verbose= False)

        return model



    def NN_s(self,camada2):


        model = tf.keras.models.Sequential()
        model.add(Dense(10,activation = "relu",input_dim = self.X_train.shape[1]))
        model.add(Dense(camada2,activation = "relu"))
        model.add(Dense(5,activation = "relu"))
        model.add(Dense(1,activation = "sigmoid"))
        #opt = tf.keras.optimizers.SGD(lr = 0.001)
        opt = tf.keras.optimizers.Adam()
        model.compile(optimizer = opt,loss = 'binary_crossentropy',metrics=  ['accuracy'])
        model.fit(self.X_train.astype(float),self.y_train.astype(int),epochs = 300,verbose= False)

        return model        





    def model_ff(self,epochs = 500):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, input_shape=(self.X_train.shape[1],), activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1,activation = 'sigmoid') 
        ])

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=250, verbose=1, mode='auto', restore_best_weights=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=  ['accuracy'])
        history = model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=epochs,callbacks = [es])

        return history


    def model_conv(self):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(target_shape=[216, 1], input_shape=(216,)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1)
    ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        history = model.fit(self.X_train, self.y_train,  epochs=1000, batch_size=32)
        return history






