import pandas as pd

def procesar_csv_ventas(ruta_archivo: str) -> pd.DataFrame:

    #Carga y limpia un archivo CSV corporativo aplicando estándares de ingeniería de datos.
    
    print(f"[INFO] Leyendo archivo desde: {ruta_archivo}")
    
    # 1. Carga segura del CSV especificando la codificación utf-8
    try:
        df = pd.read_csv(ruta_archivo, encoding='utf-8') 
    except Exception:
        # Fallback por si fue guardado con otra codificación típica de Windows
        df = pd.read_csv(ruta_archivo, encoding='latin1')
        #utf-8 es la codificacion mas comun para archivos csv, si no funciona se puede cambiar a latin1

    print(f"[INFO] Datos cargados con éxito. Dimensiones iniciales: {df.shape}")

    # 2. Eliminación de columnas irrelevantes (Ruido / Identificadores únicos)
    # 'id_venta' no aporta patrón predictivo, solo identifica la fila.
    columnas_a_borrar = ['id_venta', 'cliente'] #aqui estan las columnas a eliminar, si queremos eliminar otra solo agregala
    for col in columnas_a_borrar: #col significa columna, es una variable temporal para iterar sobre la lista de columnas a borrar
        if col in df.columns: #si la variable temporal col esta en las columnas del dataframe se ejecuta el siguiente bloque
            df = df.drop(columns=[col]) #aqui definimos que se borra la columna asignada a la iteracion con col
            print(f"[INFO] Columna irrelevante '{col}' eliminada.") #imprimos la iteracion col que dentro de ella esta la columna eliminada

    # 3. Limpieza de fechas (Pasar de texto plano a formato temporal datetime de pandas)
    if 'fecha' in df.columns: #si fecha esta en lasc columnas del dataframe:
        df['fecha'] = pd.to_datetime(df['fecha']) #convertimos la columna fecha a formato datatime
        # Truco Senior: Extraer componentes útiles de la fecha para el modelo dividiendo en año, mes y dia
        df['año'] = df['fecha'].dt.year #año es igual a estar en la columna fecha y lo extraemos con dt.year (dt es datatime)
        df['mes'] = df['fecha'].dt.month #hacemos lo mismo con mes y dia
        df['dia'] = df['fecha'].dt.day
        df = df.drop(columns=['fecha']) # Borramos la fecha cruda
        print("[INFO] Columna 'fecha' desglosada en año, mes y día.")

    print(f"[INFO] Dimensiones finales tras la limpieza: {df.shape}")

    #df.select_dtypes(include=['object'])
    #imprimimos el df seleccionando el tipo de dato object, que son los datos categoricos
    
    #trasformamos la columna producto a numeros, para que el modelo pueda trabajar tambien con ella:
    #df = pd.get_dummies(df, columns=['producto'], drop_first=True) #drop_first=True para evitar la trampa de la variable ficticia
    
    #si quieres asignarle a cada producto un numero (como 0, 1, 2) podemos usar un label encoding:
    #df['producto'] = df['producto'].astype('category').cat.codes 
    #esto asigna un numero a cada producto, empezando desde 0

    return df

if __name__ == "__main__":
    # Cambia 'practicas_sql_ventas2.csv' por la ruta exacta de tu archivo exportado desde MySQL
    # (Asegúrate de exportarlo como CSV desde Workbench si aún no lo tienes guardado como archivo plano).
    archivo_csv = "practicas_sql_ventas2.csv"
    
    df_limpio = procesar_csv_ventas(archivo_csv)
    print(df_limpio.head(5))