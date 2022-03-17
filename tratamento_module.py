from pyexpat.errors import XML_ERROR_NOT_STANDALONE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
class Tratamento:

    def __init__(self):
        self.path_positivo =  'Dados/compilado_positivo.xlsx'
        self.path_negativo = 'Dados/compilado_negativo.xlsx'



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

    def get_samples(self,amostras):
        self.X = amostras.drop('Diagnostico',axis=1).copy()
        self.y = amostras['Diagnostico'].copy()
        self.y = self.y.replace(('Positivo','Negativo'),('1','0'))

        return self.X,self.y

    def training_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size = 0.1,random_state = 42)
        
        return X_train,X_test,y_train,y_test

    def confusion(self,y_true,y_predict):
        cm = confusion_matrix(y_true,y_predict)
        class_names= ['Negativo','Positivo']
        fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens,
                                        class_names = class_names)
        plt.xlabel('Prredições', fontsize=18)
        plt.ylabel('Verdadeiros', fontsize=18)
        plt.title('Matrix de confusão', fontsize=18)
        plt.show()


