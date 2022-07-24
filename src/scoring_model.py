import pandas as pd
import pickle
import os
import numpy as np
from sklearn.linear_model import LinearRegression

#Metricas de evaluación
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import r2_score


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) / y_true) * 100



# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../data/raw', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['ingres'],axis=1)
    y_test = df[['ingres']]
    y_pred_test=model.predict(X_test)
    
    
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred_test))
    print('MSE:', metrics.mean_squared_error(y_test, y_pred_test))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))
    print('MAPE:', mean_absolute_percentage_error(y_test, y_pred_test))

    
# Entrenamiento completo
def main():
    df = eval_model('data_train.csv')
    print('Finalizó la validación del Modelo')

if __name__ == "__main__":
    main()