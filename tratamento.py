import numpy as np
import pandas as pd
import os


class Tratamento(self):

    def __init__(self):
        self.path_positivo =  'Dados/compilado_positivo.xlsx'
        self.path_negativo = 'Dados/compilado_negativo.xlsx'



    def amostras_import(self):
        amostras_p = pd.read_excel(self.path_positivo,engine='openpyxl')
        amostras_p.set_index('Wavelength',inplace=True)
        amostras_p = amostras_p.T
        amostras_p['Diagnostico'] = 'Positivo'
        amostras_n = pd.read_excel(self.path_negativo,engine='openpyxl')
        amostras_n.set_index('Wavelength',inplace=True)
        amostras_n = amostras_n.T
        amostras_n['Diagnostico'] = 'Negativo'
        amostras = amostras_p.append(amostras_n)
        
        return amostras


