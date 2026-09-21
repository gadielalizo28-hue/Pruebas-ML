import torch
import torch.nn as nn
import torch.nn.functional as F #importamos las funciones matematicas que no tienen parametros internos, esto funciona como un modulo de funciones

class CNNProcesadora(nn.Module):
    """
    Arquitectura CNN modular orientada a objetos para clasificacion de imagenes.
    Sigue los estandares de producccion de pytorch.
    """
    def __init__(self, num_classes: int = 10): #el num_classes es un parametro con un valor por defecto, son cuantas categorias va a clasificar
        super(CNNProcesadora, self).__init__()

        #bloque convolucional 1: Extrae caracteristicas de bajo nivel (bordes, esquinas)
        #Recibe una imagen de 3 canales puede ser de 1 para ecscala gris o a 4 con multiples colores (RGB) 
        # y saca 16 mapas de caracteristicas, empiezas primero con 16 y avanzando hasta 512
        #kernel size: cada filtro es una matris de 3x3 pixeles, rara vez puedes usar 7x7
        #stride: el filtro se desplaza de 1 en 1 pixel a la vez
        #el padding añade un borde de 1 pixel vacio al rededor de la imagen. Esto combina todo perfectamente
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        #Bloque convolucional 2: Extrae caracteristicas de nivel medio (formas, texturas)
        #in channels tiene que coincidir exactamente con el out channels para recibir las que el conv1 proceso
        #out channels ahora aplicaremos 32 filtros nuevos aumentando la profundidad y mantenemos los otros 3 parametros
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        #Bloque convolucional 3: Extrae caracteristicas de alto nivel (objetos, patrones complejos)
        #lo mismo aqui el in mantiene el numero del out anterior y mantenos los otros 3 parametros iguales
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        #bloque convolucional 4:
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) 

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #reduce las dimensiones espcaciales a la mitad

        #Capas densas(fully connected): Realizan la clasificacion final
        #asumiendo una imagen de entrada de 64x64 pixeles que se reduce tras el pooling
        self.fc1 = nn.Linear(128 * 4 * 4, 256) #256 es el numero de neuronas ocultas lo mismo en fc2
        #al principio era 64x64 la imagen, cayo por la primera capa 32x32, luego 16x16, luegoo a 8x8 y finalmente 4x4
        self.fc2 = nn.Linear(256, num_classes) #num_classes es el numero de clases a clasificar
        
        #dropout para prevenir overfitting:
        #En el entrenamiento apagara aleatoriamente el 50% de las neuronas en cada paso, esto obliga a la red a no
        #depender de una sola neurona y a aprender patrones mas robustos
        self.dropout = nn.Dropout(p=0.5) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #1. Primera capa conv + ReLu + reduccion espacial (pooling)
        x = self.pool(F.relu(self.conv1(x))) #aplica conv1, ReLU y luego max pooling
        #el tensor x entra en conv1, el resultado pasa por la funcion F.ReLu y finalmente pasa por el pooling

        #2. Segunda capa conv + ReLu + reduccion espacial (pooling)
        x = self.pool(F.relu(self.conv2(x))) 

        #3. Tercera capa conv + ReLu + reduccion espacial (pooling)
        x = self.pool(F.relu(self.conv3(x)))

        x = self.pool(F.relu(self.conv4(x)))

        #4. Aplanamiento (Flatten) para pasar de matriz/tensor 3D a vector 1D
        x = x.view(x.size(0), -1) #redimensiona el tensor para la capa fully connected
        
        #5. Capas densar (fully connected) con ReLu y dropout
        x = F.relu(self.fc1(x)) #el vector pasa por la primera capa densa y aplica el ReLu
        x = self.dropout(x) #aplica dropout en el vector x
        x = self.fc2(x) #aplica fc2 para obtener las predicciones finales

        return x

if __name__ == "__main__":
    #Prueba rapida del forward pass con un tensor simulado (batch de 4 imagenes RGB 32x32)
    tensor_prueba = torch.randn(4, 3, 64, 64)
    modelo = CNNProcesadora(num_classes=10)
    salida = modelo(tensor_prueba)

    print(f"[EXITO] Forward pass completado.")
    print(f"[INFO] Forma del tensor de entrada: {tensor_prueba.shape}")
    print(f"[INFO] Forma del tensor de salida: {salida.shape}")