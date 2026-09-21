import torch
import torch.nn as nn
from model import RedProfesional #importamos nuestro modelo

def iniciar_entrenamiento() -> None:
    #1.- Configuracion de parametros corporativos (variables)
    BATCH_SIZE = 8
    INPUT_DIM = 20
    HIDDEN_DIM = 32
    OUTPUT_DIM = 4
    LEARNING_RATE = 0.01
    EPOCAS = 2

    #2.- Seleccion estricta de Hardware (fijo)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Servidor Inicializadop. Hardware activo: {dispositivo}")

    #bloque de seguridad empreserial
    try:
        modelo = RedProfesional(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
        modelo = modelo.to(dispositivo)

        criterio = nn.CrossEntropyLoss()
        optimizador = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE)

        print("[INFO] Pipeline Copilado. Iniciando Ciclos de entrenamiento...")

        for epoca in range(EPOCAS):
            #simulamos los datos de entrada corporativos
            datos_entrada = torch.randn(BATCH_SIZE, INPUT_DIM, device=dispositivo)
            etiquetas_reales = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=dispositivo)

            #ciclo de aprendizaje estandar
            predicciones = modelo(datos_entrada)
            perdida = criterio(predicciones, etiquetas_reales)

            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()

            print(f"-> Epoca [{epoca +1}/{EPOCAS}] completada | Perdida: {perdida.item():.4f}")
        
        print("[EXITO] Proceso Finalizado sin Errores en Produccion.")

    except Exception as e:
        #Manejo limpio ante desastres en el servidor
        print(f"[ERROR CRITICO] Fallo el pipeline de entrenamiento: {str(e)}")
    
if __name__ == "__main__":
    iniciar_entrenamiento()