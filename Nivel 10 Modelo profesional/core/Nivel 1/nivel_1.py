import torch
import torch.nn as nn

# 1. Definimos nuestro dispositivo (GPU o CPU, tal como lo hiciste en tus fotos)
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Seleccionado: {dispositivo}")

class RedMasterNivel1(nn.Module):
    """
    Arquitectura de una Red Multicapa (MLP) utilizando capas lineales estándar.
    Aquí combinamos múltiples capas para ver cómo los datos fluyen hacia adelante.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # Capa Oculta 1: Recibe las entradas y las proyecta a un espacio oculto
        self.capa_oculta = nn.Linear(input_dim, hidden_dim)
        self.capa_oculta2 = nn.Linear(hidden_dim, hidden_dim)
        self.capa_oculta3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Función de activación: Rompe la linealidad (permite que la red aprenda patrones complejos)
        self.activacion = nn.ReLU()
        
        # Capa de Salida: Proyecta el espacio oculto hacia el resultado final esperado
        self.capa_salida = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        El Forward Pass (Paso hacia adelante): 
        Cómo viajan los datos desde que entran hasta que la red da una respuesta.
        """
        #modificando la secuencia del peso 1 y 2
        z1 = self.capa_oculta(x)
        a1 = self.activacion(z1)
    
        z2 = self.capa_oculta2(a1) #la segunda capa toma la activacion de la primera
        a2 = self.activacion(z2)

        z3 = self.capa_oculta3(a2)
        a3 = self.activacion(z3)

        # Paso 3: la ultima capa recibe solo el resultado final
        salida = self.capa_salida(a3) 
        
        return salida

# --- Comprobando el funcionamiento ---
if __name__ == "__main__":
    # Dimensiones de prueba
    BATCH_SIZE = 16      # 16 es el numero de cuantos datos quieras procesar
    INPUT_FEATURES = 20 # 20 depende mucho si tienes un csv con 10 columnas pues ese es el numero
    HIDDEN_UNITS = 32   # 32 esto es depende de cuan compleja quieres que sean las capas ocultas, si el numero es mas grande entonces es mas lenta (16, 32, 64, 128)
    CLASSES = 4         # 4 te lo dicta el problema segun lo que buscas, si estas buscando entre 2 parametros las clases son 2, si estas clasificando digitos del 0 al 9 son 10 clases

    # Instanciamos el modelo y lo mandamos al hardware correcto
    modelo = RedMasterNivel1(input_dim=INPUT_FEATURES, hidden_dim=HIDDEN_UNITS, output_dim=CLASSES)
    modelo = modelo.to(dispositivo)

    # Creamos un lote de datos simulados (como las bandejas de tu DataLoader)
    lote_datos_x = torch.randn(BATCH_SIZE, INPUT_FEATURES, device=dispositivo)

    # Ejecutamos el Forward Pass
    predicciones = modelo(lote_datos_x)

    print("\n--- Resultados del Forward Pass (Nivel 1 adaptado a tu estilo) ---")
    print(f"Dimensiones de la entrada (lote): {lote_datos_x.shape}")
    print(f"Dimensiones de la salida de la red: {predicciones.shape}")
    print(f"Predicciones generadas:\n{predicciones}")

    #validacion de dimensiones:
    assert predicciones.shape == (BATCH_SIZE, CLASSES), f"Error: Dimensiones incorrectas{predicciones.shape}"
    print(f"\033[92m!Exito! las dimensiones de salida coinciden correctamente y la red probada con {CLASSES} clases de salida.\033[0m") #033 es un numero especial para el color verde
    