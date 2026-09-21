import torch
import torch.nn as nn

class ProcesadorSecuencialNLP(nn.Module):
    """
    Arquitectura basada en LSTM para procesamiento de texto y secuencias temporales.
    Ideal para traducción base, análisis de sentimiento o predicción de valores en el tiempo.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_clases: int):
        #vocab_size = es el tamaño del vocabulario
        #embed_dim = el tamaño de los vectores de palabras
        #hidden_dim = la memoria interna de la LSTM
        #num_clases = cuantas categorias vas a predecir y todos esos parametros como int (numeros)
        super(ProcesadorSecuencialNLP, self).__init__()
        
        # 1. Capa de Embeddings: Transforma índices de palabras en vectores densos con significado semántico
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        #embedding es un comando especial para las palabras, si quisieramos predecir numeros tendriamos que cambiar esta capa por una lineal (nn.Linear)
        #num_embeddings=vocab_size = cuantas palabras unicas existen en el diccionario
        #embed_dim = es el tamaño del vector continuo que representara a cada palabra 
        #(ojo que los parametros los definiremos en el ultimo bloque)

        # 2. Capa LSTM (Long Short-Term Memory): Captura dependencias secuenciales a largo plazo
        self.lstm = nn.LSTM( #inicializa la celda de memoria a largo y corto plazo
            input_size=embed_dim, #el input tiene que dar igual al embed_dim
            hidden_size=hidden_dim, #el hidden size sera igual al hidden dim (mientras mas grande mas patrones complejos puede recordar, pero consume mas memoria)
            batch_first=True, # Asegura que el batch sea la primera dimensión (batch_size, seq_len, features)
            num_layers=2,     # Apilamos 2 capas LSTM para mayor profundidad de análisis
            dropout=0.3       # Previene sobreajuste entre capas recurrentes (apaga el 30% de las conexiones)
        )
        
        # 3. Capa Densa Final (Classifier Head): Mapea el estado oculto final a las clases o salidas de la tarea
        self.fc = nn.Linear(hidden_dim, num_clases) #se aplica sobre el hidden dim y el numero de clases

    def forward(self, texto_indices: torch.Tensor) -> torch.Tensor:
        # 1. Convertir índices enteros de palabras en vectores densos -> (Batch, Seq_Len, Embed_Dim)
        x = self.embedding(texto_indices) #el texto sera convertido con embedding en texto y todo esto dentro de un vector x
        #cada ID se convierte en su vector de 128 numeros haciendo asi que x tenga 3 dimensiones como resultado

        # 2. Pasar la secuencia por la LSTM
        # LSTM retorna una tupla: (output_secuencias, (hidden_state_final, cell_state_final))
        lstm_out, (hn, cn) = self.lstm(x) 
        #los vectores x estan dentro de la funcion lstm y todo esto dentro de una variable que tiene la memoria a corto y largo plazo
        
        # 3. Tomar únicamente el estado oculto de la última capa temporal para la clasificación
        # hn[-1] extrae la última capa del tensor de estado oculto -> (Batch, Hidden_Dim)
        ultima_representacion = hn[-1] #nunca nos importa lo que la ia penso a mitad del texto si no su conclusion final
        #por eso solo extraemos la capa final apilada (en este caso capa 2)
        
        # 4. Generar logits finales
        salida = self.fc(ultima_representacion)
        
        return salida

if __name__ == "__main__":
    # Parámetros del modelo que llamamos arriba dentro del codigo y fuera del if name main
    VOCAB_SIZE = 10000  # Diccionario de 10,000 palabras únicas
    EMBED_DIM = 128     # Cada palabra se representará con 128 dimensiones de significado
    HIDDEN_DIM = 256    # Neuronas internas de la memoria LSTM
    NUM_CLASES = 3      # Clases de salida (ej. positivo, neutro, negativo)
    
    modelo_nlp = ProcesadorSecuencialNLP(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASES)
    
    print(f"[EXITO] Modelo LSTM secuencial inicializado correctamente.")
    
    # Simulación de entrada: Batch de 2 textos, cada uno compuesto por una secuencia de 20 palabras (índices enteros)
    tensor_texto_prueba = torch.randint(0, VOCAB_SIZE, (2, 20)) 
    #generamos datos falsos sobre la prueba (entre el 0 y 10000) con el tamaño de 2 textos y 20 palabras
    salida = modelo_nlp(tensor_texto_prueba)
    
    print(f"[INFO] Forma del tensor de entrada (índices de palabras): {tensor_texto_prueba.shape}")
    print(f"[INFO] Forma del tensor de salida (Logits): {salida.shape}")

    """
    Cosas importantes a recalcar:

    ¿Cómo saber qué componentes VAN y cuáles NO VAN?
    Todo depende de la forma de tu entrada y la forma de tu salida. La arquitectura debe moldearse para conectar ambos 
    extremos de forma 
    
    lógica:¿La entrada es texto o categorías secuenciales? 
    VA obligatoriamente una capa de nn.Embedding. Las palabras necesitan convertirse a vectores.
    
    ¿La entrada son imágenes? NO VA una LSTM ni un Embedding al inicio. En su lugar, van Capas Convolucionales (nn.Conv2d) 
    para extraer bordes y formas.
    
    ¿El orden de los datos importa (series de tiempo, texto, audio)? VA una capa secuencial como nn.LSTM, nn.GRU o un 
    bloque Transformer.
    
    ¿El orden de los datos NO importa (tablas de Excel con datos fijos como edad, salario, etc.)? NO VA una LSTM. 
    Solo usas capas lineales/densas (nn.Linear) apiladas
    """