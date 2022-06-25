from ctypes.wintypes import BOOLEAN
from pyexpat.errors import XML_ERROR_NOT_STANDALONE
from turtle import color
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import imblearn
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
from scipy.signal import savgol_filter





class Tratamento:

    def __init__(self):
        self.path_positivo =  'Dados/compilado_positivo.xlsx'
        self.path_negativo = 'Dados/compilado_negativo.xlsx'
        self.path_figuras = 'Figuras/'



    def amostras_import(self):
        amostras_p = pd.read_excel(self.path_positivo,engine='openpyxl')
        amostras_p = amostras_p.dropna()
        amostras_p.set_index('Wavelength',inplace=True)
        amostras_p = amostras_p.T
        amostras_p['Diagnostico'] = 'Positivo'
        amostras_n = pd.read_excel(self.path_negativo,engine='openpyxl')
        amostras_n= amostras_n.dropna()
        amostras_n.set_index('Wavelength',inplace=True)
        amostras_n = amostras_n.T
        amostras_n['Diagnostico'] = 'Negativo'
        amostras = amostras_p.append(amostras_n)
        amostras = amostras[~(amostras.drop(columns=['Diagnostico'])> 1).any(1)]
        #amostras.columns = amostras.columns.str.replace(' ','_')
        #amostras.columns = amostras.columns.str.replace('.0','')
        return amostras

    def get_frequencies(self,amostras, freq: list = None):
        if freq != None:
            amostras = amostras[freq]
        return amostras

    def get_frequencies_from(self,amostras,freq):
        if freq > 350:
            freq = freq-350
        else:
            return
        columns = amostras.drop(columns='Diagnostico').columns.to_numpy()
        columns = columns[freq:,]
        columns = np.append(columns,'Diagnostico')

        amostras = amostras[columns]


        
        return amostras
 



    def get_samples(self,amostras):
        self.X = amostras.drop('Diagnostico',axis=1).copy()
        self.y = amostras['Diagnostico'].copy()
        self.y = self.y.replace(('Positivo','Negativo'),('1','0'))

        return self.X,self.y


    def training_test(self,X ,y ):


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size = 0.3,random_state = 37)

        return self.X_train,self.X_test,self.y_train,self.y_test



    def under_sample(self,X_train =  None,y_train = None,sampling_strategey = "majority"):
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train

        
        
        undersampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy = sampling_strategey)
        self.X_under, self.y_under = undersampler.fit_resample(self.X_train,self.y_train)

        return self.X_under, self.y_under

    def over_sample(self,X_train = None, y_train = None):
        if X_train is None:
            X_train = self.X_train
        if  y_train is None:
            y_train = self.y_train

        
        oversampler = imblearn.over_sampling.RandomOverSampler()
        self.X_over, self.y_over = oversampler.fit_resample(X_train,y_train)

        return self.X_over,self.y_over
        


    def PCA_t(self,X_train,n_components = 4):
        pca = PCA(n_components = n_components)
        if isinstance(X_train, pd.DataFrame):
            pca.fit(X_train.to_numpy())
        else:
            pca.fit(X_train)
        

        return pca
        

    def savgol(X_values, window_lenght = 11, polyorder =5, deriv = 1):

        x_sav = savgol_filter(x=X_values,window_length=window_lenght,polyorder=polyorder,deriv=deriv)

        return x_sav


    def confusion(self,y_true,y_predict,fig_name ="confusion.png" ):
        cm = confusion_matrix(y_true,y_predict)
        class_names= ['Negativo','Positivo']
        fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), cmap=plt.cm.Greens,
                                        class_names = class_names)
        ax
        plt.xlabel('Predições', fontsize=12)
        plt.ylabel('Verdadeiros', fontsize=12)
        plt.title('Matriz de confusão', fontsize=12)
        plt.savefig(self.path_figuras+ fig_name, bbox_inches = 'tight')
        plt.show()
        
        return

    def precisionrecall(self,y_true,y_predict,name = "SVC"):

        display = PrecisionRecallDisplay.from_estimator(
            classifier, X_test, y_test, name=name
        )
        _ = display.ax_.set_title("2-class Precision-Recall curve")

    def roc(self,y,pred,estimator_name: str = 'None'):
        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        roc_auc = metrics.auc(fpr, tpr)
 
        
        return fpr,tpr,thresholds,roc_auc

    
    def plot_roc(self,fpr: list, tpr: list, Names: list, name_curve: str, auc: list, tpr_threshold = 0.8,):
        colors = ['r','b']
        plt.figure(figsize = (8,8))
        for i in range(len(fpr)):
            plt.plot(fpr[i], tpr[i], colors[i],label=Names[i] + " Área: " + str("{:.2f}".format(auc[i])))
       
        plt.legend(loc="lower right")
        plt.title("Curva ROC RNA" + " "+ name_curve, fontsize = 15)
        plt.xlabel("Falsos Positivos",fontsize = 15)
        plt.ylabel("Verdadeiros Positivos", fontsize = 15)
        plt.grid()
        plt.savefig(self.path_figuras + name_curve+'.png', bbox_inches = 'tight')
        plt.show()

