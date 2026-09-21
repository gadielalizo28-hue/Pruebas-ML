import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_pro2 import RedProfesional  # Reutilizamos el modelo definido en otro archivo
from dataset_pro import DatasetMedicoReal  # Reutilizamos el dataset definido en otro archivo

# La función principal se encarga de cargar datos, crear el modelo,
# calcular la pérdida, hacer backpropagation y ajustar los pesos.
def entrenar_con_datos_reales() -> None:
    # 1) Hiperparámetros del entrenamiento
    # BATCH_SIZE: cuántas muestras se procesan juntas en cada paso.
    BATCH_SIZE = 32 
    #si colocaramos un numero mas pequeño la maquina tendria menos lotes en los cuales dividir la informacion, lo cual hace un aprendizaje mas agresivo 

    # INPUT_DIM: número de características de cada ejemplo.
    # En este caso, el dataset médico tiene 30 columnas/variables.
    INPUT_DIM = 30

    # HIDDEN_DIM: tamaño de la capa oculta de la red.
    HIDDEN_DIM = 64

    # OUTPUT_DIM: número de clases de salida.
    # Como es un problema binario, hay 2 salidas posibles.
    OUTPUT_DIM = 2

    # LEARNING_RATE: cuanto se ajustan los pesos en cada paso.
    LEARNING_RATE = 0.006

    # EPOCAS: cuántas veces se recorre todo el conjunto de datos durante el entrenamiento.
    EPOCAS = 30

    # 2) Selección de hardware
    # Si hay GPU disponible, se usa CUDA; si no, se usa CPU.
    # Esto sirve para que el entrenamiento sea más rápido si hay una tarjeta gráfica.
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Servidor listo. Hardware activo: {dispositivo}")

    # try/except: si algo falla, se captura el error y se imprime.
    try:
        # 3) Creación del dataset real y del DataLoader
        # DatasetMedicoReal(): prepara los datos médicos reales.
        dataset_real = DatasetMedicoReal()

        # DataLoader organiza los datos en lotes y los mezcla aleatoriamente.
        # shuffle=True ayuda a que el modelo no aprenda en un orden fijo.
        dataloader_pro = DataLoader(dataset_real, batch_size=BATCH_SIZE, shuffle=True)

        # 4) Crear modelo, función de pérdida y optimizador
        # RedProfesional es el modelo neuronal que importamos desde model_pro2.py.
        modelo = RedProfesional(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)

        # Mover el modelo al dispositivo correcto (GPU o CPU).
        modelo = modelo.to(dispositivo)

        # CrossEntropyLoss: función de pérdida utilizada para clasificación.
        # En un problema con 2 clases, funciona bien.
        criterio = nn.CrossEntropyLoss()

        # AdamW: optimizador que ajusta los pesos del modelo para reducir la pérdida.
        optimizador = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE)

        print("[INFO] Pipeline con datos reales inicializado. Entrenando...")

        # 5) Bucle principal de entrenamiento
        # Recorremos cada época (cada pasada completa sobre todo el dataset).
        for epoca in range(EPOCAS):
            # modelo.train(): activamos el modo de entrenamiento.
            # Esto es importante porque algunas capas (como Dropout) se comportan distinto
            # en entrenamiento que en validación.
            modelo.train()

            # pérdida_acumulada: sumamos la pérdida de todos los lotes para hacer un promedio
            # al final de la época.
            perdida_acumulada = 0.0

            # Iteramos lote por lote usando el DataLoader.
            # En cada iteración, lote_x son las entradas y lote_y son las etiquetas correctas.
            for lote_x, lote_y in dataloader_pro:
                # Mover los tensores al mismo hardware que el modelo.
                lote_x = lote_x.to(dispositivo)
                lote_y = lote_y.to(dispositivo)

                # Pasada hacia adelante:
                # El modelo recibe las características del lote y genera predicciones.
                predicciones = modelo(lote_x)

                # Calculamos cuán equivocadas fueron esas predicciones.
                perdida = criterio(predicciones, lote_y)

                # Limpiamos gradientes anteriores antes de calcular los nuevos.
                # Esto es necesario porque PyTorch acumula gradientes.
                optimizador.zero_grad()

                # backward(): calcula los gradientes de la pérdida respecto a los pesos.
                perdida.backward()

                # clip_grad_norm_: evita que los gradientes crezcan demasiado.
                # Esto ayuda a evitar "explosión de gradientes". esto es siempre necesario
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)

                # Actualizamos los pesos del modelo con el optimizador.
                optimizador.step()

                # Acumulamos la pérdida actual para luego hacer el promedio final de la época.
                perdida_acumulada += perdida.item()

            # Calculamos la pérdida promedio de la época.
            # Se divide entre la cantidad de lotes.
            perdida_promedio = perdida_acumulada / len(dataloader_pro)

            # Mostramos el progreso por época.
            print(f"-> Época [{epoca + 1}/{EPOCAS}] completada | Pérdida Real Promedio: {perdida_promedio:.4f}")

        # Mensaje final si el entrenamiento termina sin errores.
        print("[EXITO] ¡Modelo entrenado con datos médicos reales y DataLoader profesional sin errores!")

    # Si ocurre algún error durante el entrenamiento, lo mostramos.
    except Exception as e:
        print(f"[ERROR CRITICO] Falló el pipeline de datos reales: {str(e)}")

# Este bloque asegura que la función se ejecute solo cuando ejecutas este archivo directamente.
# Si importas este archivo desde otro archivo, no se ejecutará automáticamente.
if __name__ == "__main__":
    entrenar_con_datos_reales()