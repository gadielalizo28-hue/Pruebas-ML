import pandas as pd

def procesar_csv_ventas(ruta_archivo: str) -> pd.DataFrame:
    
    print(f"[INFO] Leyendo archivo desde: {ruta_archivo}")
    
    try:
        df = pd.read_csv(ruta_archivo, sep=';', encoding='utf-8') 
    except Exception:
        df = pd.read_csv(ruta_archivo, sep=';', encoding='latin1')

    print(f"[INFO] Datos cargados con éxito. Dimensiones iniciales: {df.shape}")

    columnas_a_borrar = ['id_venta', 'cliente', 'fecha'] 
    for col in columnas_a_borrar: 
        if col in df.columns: 
            df = df.drop(columns=[col])
            print(f"[INFO] Columna irrelevante '{col}' eliminada.")


    print(f"[INFO] Dimensiones finales tras la limpieza: {df.shape}")

    return df

if __name__ == "__main__":
    archivo_csv = "practicas_sql_ventas3.csv"
    
    df_limpio = procesar_csv_ventas(archivo_csv)
    print(df_limpio.head(5))

    