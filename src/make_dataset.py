# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
##import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Preprocesar label
from sklearn import preprocessing
# Estandarizamos los datos
from sklearn.preprocessing import StandardScaler
# Normalizando los datos
from sklearn.preprocessing import MinMaxScaler


# División de data entrenamiento y test
from sklearn.model_selection import train_test_split



# Leemos los archivos csv
def read_file_excel(filename):
    df = pd.read_excel(os.path.join('../data/raw/', filename), sheet_name = "demomodifmodern2", header = 0)
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):
    X = df.drop(['ingres'], axis=1)
    y = df[['ingres']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
    dataset1 = X_train
    dataset1['ingres']=y_train['ingres']
    
    #Métodos de detección de valores atípicos multivariante

    #Se utilizará las variables
    cols = ['ingres','Gastocoche','Aniosempleo']
    
    #Isolation Forests

    clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01, \
                        max_features=3, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
    clf.fit(dataset1[cols])
    pred = clf.predict(dataset1[cols])
    dataset1['anomaly']=pred
    outliers_if=dataset1.loc[dataset1['anomaly']==-1]
    outlier_if_index=list(outliers_if.index)
    #print(outlier_index)
    #print(dataset1['anomaly'].value_counts())
    
    #Métodos de detección de valores atípicos multivariante

    #Se utilizará las variables
    cols = ['ingres','edad','AniosDireccion']
    
    #Isolation Forests
    dataset1_filtro1=dataset1.loc[dataset1['anomaly']==1].drop(['anomaly'], axis=1)


    clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=0.02, \
                        max_features=3, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
    clf.fit(dataset1_filtro1[cols])
    pred = clf.predict(dataset1_filtro1[cols])
    dataset1_filtro1['anomaly_2']=pred
    outliers_if=dataset1_filtro1.loc[dataset1_filtro1['anomaly_2']==-1]
    outlier_if_index=list(outliers_if.index)
    #print(outlier_index)
    
    #print(dataset1_filtro1['anomaly_2'].value_counts())
    
    dataset1_sinOutliers_vr2=dataset1_filtro1.loc[dataset1_filtro1['anomaly_2']==1].drop(['anomaly_2'], axis=1)
    #dataset1.info()
    #dataset1_sinOutliers_vr2.info()
    
    X_train_2 = dataset1_sinOutliers_vr2.drop(['ingres'], axis=1)
    y_train_2 = dataset1_sinOutliers_vr2[['ingres']]
    
    data_train = X_train_2
    data_train['ingres'] =  y_train_2
    
    data_test = X_test
    data_test['ingres'] = y_test
    
    return data_train, data_test


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, filename):
    df.to_csv(os.path.join('../data/raw/', filename))
    print(filename, 'exportado correctamente en la carpeta raw')
    
    

# Generamos las matrices de datos que se necesitan para la implementación

def main():
    
    df = read_file_excel('rawdata-DataInferenciaIngresos.xlsx')
    data_train, data_test = data_preparation(df)
    
    # Matriz de Entrenamiento
    data_exporting(data_train, 'data_train.csv')
    # Matriz de Validación
    
    data_exporting(data_test, 'data_test.csv')
    
if __name__ == "__main__":
    main()
    
    
    