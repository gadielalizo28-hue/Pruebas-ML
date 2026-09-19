import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer #el simple imputer sirve para rellenar los valores nulos de las columnas con la media
import lightgbm as lgb #el lgbm es un modelo que tiene la capacidad de mirar los datos nulos y decidir si los ignora o los usa, es un modelo muy potente para datos tabulares
from sklearn.metrics import mean_squared_error

def entrenar_lightgbm_con_nulos(ruta_csv: str) -> None:
    # 1. Cargamos el CSV corporativo
    df = pd.read_csv(ruta_csv, sep=';', encoding='utf-8')
    
    # Separamos características (X) y objetivo (y)
    X = df.drop(columns=['total', 'id_venta', 'cliente', 'fecha'], errors='ignore')
    y = df['total']

    # 2. División de datos (Train/Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Tratamiento Senior de Nulos: Imputación numérica
    # Si hay celdas vacías en números, las rellenamos con la mediana para no alterar la tendencia.
    imputador = SimpleImputer(strategy='median') 

    #tambien puedes usar estas estrategias: 
    #mean rellena con el promedio, median rellena con la mediana, most_frequent rellena con el valor mas frecuente, constant rellena con un valor constante que tu definas
    
    # Identificamos columnas numéricas y categóricas
    cols_num = X_train.select_dtypes(include=['int64', 'float64']).columns

    # Aplicamos la imputación solo a las columnas numéricas
    X_train[cols_num] = imputador.fit_transform(X_train[cols_num])
    X_test[cols_num] = imputador.transform(X_test[cols_num])

    # Convertimos texto categórico a tipo 'category' para que LightGBM lo procese nativamente
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = X_train[col].astype('category') #astype('category') convierte la columna a tipo categoria, que es un tipo de dato que entiende lightgbm
        X_test[col] = X_test[col].astype('category')

    # 4. Creación y entrenamiento del modelo LightGBM
    modelo = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    modelo.fit(X_train, y_train)

    # 5. Evaluación
    predicciones = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predicciones)
    print(f"[EXITO] LightGBM entrenado con éxito. MSE: {mse:.4f}")

if __name__ == "__main__":
    entrenar_lightgbm_con_nulos("practicas_sql_ventas3.csv")
    pass