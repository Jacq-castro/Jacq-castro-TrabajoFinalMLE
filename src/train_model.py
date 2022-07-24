### SCRIPT  2 train_model

import pickle
import os
from sklearn.linear_model import LinearRegression


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw', filename))
    X_train = df.drop(['ingres'],axis=1)
    y_train = df[['ingres']]
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo 
    lm = LinearRegression()
    lm.fit(X_train,y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(lm, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')
    
 


# Entrenamiento completo
def main():
    read_file_csv('data_test.csv')
    print('Finalizó el entrenamiento del Modelo')
    
 
if __name__ == "__main__":
    main()