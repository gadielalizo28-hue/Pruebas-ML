import torch
import torch.nn as nn
from model_pro import RedProfesional 

def iniciar_entrenamiento() -> None:
    BATCH_SIZE = 16
    INPUT_DIM = 10
    HIDDEN_DIM = 64
    OUTPUT_DIM = 3
    LEARNING_RATE = 0.01
    EPOCAS = 5

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Servidor Inicializadop. Hardware activo: {dispositivo}")

    try:
        modelo = RedProfesional(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
        modelo = modelo.to(dispositivo)

        criterio = nn.CrossEntropyLoss()
        optimizador = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE)

        print("[INFO] Pipeline Copilado. Iniciando Ciclos de entrenamiento...")

        for epoca in range(EPOCAS):
            datos_entrada = torch.randn(BATCH_SIZE, INPUT_DIM, device=dispositivo)
            etiquetas_reales = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), device=dispositivo)

            predicciones = modelo(datos_entrada)
            perdida = criterio(predicciones, etiquetas_reales)

            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()

            print(f"-> Epoca [{epoca +1}/{EPOCAS}] completada | Perdida: {perdida.item():.4f}")
        
        print("[EXITO] Proceso Finalizado sin Errores en Produccion.")

    except Exception as e:
        print(f"[ERROR CRITICO] Fallo el pipeline de entrenamiento: {str(e)}")
    
if __name__ == "__main__":
    iniciar_entrenamiento()