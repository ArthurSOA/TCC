import numpy as np
import pandas as pd

def amostras_import():
    amostras_p = pd.read_excel("compilado_positivo.xlsx")
    amostras_p.set_index('Wavelength',inplace=True)
    amostras_p = amostras_p.T
    amostras_p['Diagnostico'] = 'Positivo'
    amostras_n = pd.read_excel("compilado_negativo.xlsx")
    amostras_n.set_index('Wavelength',inplace=True)
    amostras_n = amostras_n.T
    amostras_n['Diagnostico'] = 'Negativo'
    amostras = amostras_p.append(amostras_n)
    return amostras


