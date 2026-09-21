from mis_archivos_n4 import procesar_csv_ventas
from sklearn.model_selection import train_test_split #es una herramienta para dividir los datos
from sklearn.compose import ColumnTransformer #es una herramienta para aplicar transformaciones especificas a columnas especificas
from sklearn.preprocessing import StandardScaler, OneHotEncoder #standardScaler es para escalar los datos numericos y OneHotEncoder es para codificar las varianbles categoricas                   
from sklearn.ensemble import RandomForestRegressor #es nuestro modelo de regresion de arboles de decision
from sklearn.pipeline import Pipeline #esto nos permite crear unn flujo de trabajo que incluye preprocesamiento y modelo en un solo objeto
from sklearn.metrics import mean_squared_error #herramienta para calificar el modelo
import pandas as pd

#a continuacion importe solo unas opciones de modelos, son de la misma categoria de random forest pero diferentes
from sklearn.linear_model import LinearRegression #esto hace regresion lineal, es un modelo mas simple que random forest, pero a veces es suficiente para ciertos problemas
from sklearn.tree import DecisionTreeRegressor #esto hace regresion de arboles de decision, es un modelo mas simple que random forest, pero a veces es suficiente para ciertos problemas

def entrenar_modelo_tabular(df: pd.DataFrame) -> None:
    # 1. Definir Variable Objetivo (Target) y Características (Features)
    # Supongamos que queremos predecir el valor 'total' de la venta
    X = df.drop(columns=['total']) #toda la informacion disponible para analizar menos la columna total (claro, de el dataframe ya limpio)
    y = df['total'] #lo que queremos predecir

    # 2. División estricta de datos (Train/Test Split) antes de cualquier transformación
    #test size le dice al modelo que el 20% de los datos se usaran para pruebas y el 80% para entrenamiento
    #esto ayuda a que no memorice datos si no que aprenda (se llega a memorizar es un error llamado fuga de datos o data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[INFO] Datos divididos -> Entrenamiento: {X_train.shape[0]} filas | Prueba: {X_test.shape[0]} filas")

    # 3. Identificar columnas numéricas y categóricas automáticamente (dividimos los datos en dos tipos de columnas: numero, texto y cateorias)
    columnas_numericas = X_train.select_dtypes(include=['int64', 'float64']).columns
    columnas_categoricas = X_train.select_dtypes(include=['object', 'category', 'str']).columns

    # 4. ColumnTransformer: Aplica transformaciones específicas a cada tipo de columna
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), columnas_numericas), #cuando pasen las columnas numericas aplica el standardScaler
            ('cat', OneHotEncoder(handle_unknown='ignore'), columnas_categoricas) #cuando pasen las columnas categoricas aplica el onehotencoder
        ] #handle_unknown='ignore' truco senior: si la maquina ve un producto nuevo que jamas vio en el entrenamiento, en vez de dar error, lo ignorara
    )

    # 5. El Pipeline Corporativo: Une preprocesamiento y modelo en una sola tubería blindada
    # Scikit-learn se encargará de ajustar el escalador ÚNICAMENTE con X_train para evitar leaks.
    pipeline_corporativo = Pipeline(steps=[
        ('preprocesamiento', preprocesador),
        ('modelo', LinearRegression()) #n_estimators=100 significa que el modelo tendra 100 arboles de decision
    ]) #hacemos la variable que tendra detro el pipeline con los pasos que son conectar el preprocesador con el algoritmo de regresion
        
    # 6. Entrenamiento del Pipeline
    pipeline_corporativo.fit(X_train, y_train) #arrancamos el entrenamiento con .fit() sobre los datos de entrenamiento
    print("[EXITO] Pipeline ajustado y entrenado sin fuga de datos.")

    # 7. Evaluación rápida en el conjunto de prueba
    pred_train = pipeline_corporativo.predict(X_train)
    pred_test = pipeline_corporativo.predict(X_test)

    mse_train = mean_squared_error(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)

    print(f"[METRICA] Error Cuadrático Medio (MSE) en Train: {mse_train:.4f}")
    print(f"[METRICA] Error Cuadrático Medio (MSE) en Test: {mse_test:.4f}")

if __name__ == "__main__":

    #definimos la ruta de el archivo original
    archivo_csv = "practicas_sql_ventas2.csv"
    
    #procesamos y limpiamos los datos
    df_limpio = procesar_csv_ventas(archivo_csv)
    
    #le pasamos esta tabla limpia al modelo para que la entrena y la evalue
    entrenar_modelo_tabular(df_limpio)

    pass