from parte_1 import procesar_csv_ventas
from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.pipeline import Pipeline 
from sklearn.metrics import mean_squared_error 
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.impute import SimpleImputer
import lightgbm as lgb

def entrenar_modelo_tabular(df: pd.DataFrame) -> None:
    
    X = df.drop(columns=['total']) 
    y = df['total'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] Datos divididos -> Entrenamiento: {X_train.shape[0]} filas | Prueba: {X_test.shape[0]} filas")

    imputador = SimpleImputer(strategy='median')

    columnas_numericas = X_train.select_dtypes(include=['int64', 'float64']).columns
    columnas_categoricas = X_train.select_dtypes(include=['object', 'category', 'str']).columns

    X_train[columnas_numericas] = imputador.fit_transform(X_train[columnas_numericas])
    X_test[columnas_numericas] = imputador.transform(X_test[columnas_numericas])


    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas), 
            ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas) 
        ] 
    )

    pipeline_corporativo = Pipeline(steps=[
        ('preprocesamiento', preprocesador),
        ('modelo', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)) 
    ]) 
        
    pipeline_corporativo.fit(X_train, y_train) 
    print("[EXITO] Pipeline ajustado y entrenado sin fuga de datos.")

    pred_train = pipeline_corporativo.predict(X_train)
    pred_test = pipeline_corporativo.predict(X_test)

    mse_train = mean_squared_error(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)

    print(f"[METRICA] Error Cuadrático Medio (MSE) en Train: {mse_train:.4f}")
    print(f"[METRICA] Error Cuadrático Medio (MSE) en Test: {mse_test:.4f}")

if __name__ == "__main__":

    archivo_csv = "practicas_sql_ventas3.csv"
    df_limpio = procesar_csv_ventas(archivo_csv)
    entrenar_modelo_tabular(df_limpio)

    pass