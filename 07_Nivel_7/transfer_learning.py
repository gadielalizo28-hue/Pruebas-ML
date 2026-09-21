import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import FakeData #usado aqui para simular un dataset de prueba

def configurar_modelo_transfer_learning(num_classes: int = 2):
    """
    Carga un modelo pre entrenado (ResNet18) y adapta su cabeza de clasificacion
    """

    #1. cargar el modelo base con pesos pre entrenados oficiales de la industria
    #'weights= models.ResNet18_Weights.DEFAULT' descarga los pesos optimizados en ImageNet
    weights=models.ResNet18_Weights.DEFAULT
    modelo = models.resnet18(weights=weights) #definimos que los pesos sean iguales a los pesos dentro de una variable

    #2. Congelar todas las capas base para que sus pesos no se destrocen entrenando desde cero
    for param in modelo.parameters():
        param.requires_grad = False

    #si quisieramos solo congelar las primeras capas de la ResNet y solo las ultimas capas se adapten junto a la cabeza nueva
    """
    1. Congelar toda la red base por defecto
    for param in modelo.parameters():
        param_requires_grad = False

    2. Descongelar unicamente el ultimo bloque de ResNet (layer4) para fine-tuning profundo
    for param in modelo.layer4.parameters():
        param_requires_grad = True

    3. La cabeza de clasificacion (fc) tambien quede abierta por defecto al ser nueva
    for param in modelo.fc.parameters():
        param_requires_grad = True
    """
    
    #si en algun momento quisieras hacer lo mismo de congelar solo las primeras capas y que las ultimas obtengan ese cambio
    #tendrias que ver las capas que hay en el modelo, como el nombre o posicion, para ver eso tienes dos formas

    #1 la mas facil:
    #print(modelo)

    #2 Iterando sobre sus modulos: la mejor opcion
    for nombre, modulo in modelo.named_children():
        print(nombre)

    #3. Inspeccionar y reemplazar la cebeza clasificacion (Classifier Head)
    # ResNet original termina en una capa lineal llamada 'fc' que clasifica mil clases de ImageNet
    #obtenemos las dimensiones de entrada de esa capa ('in_features')
    in_features = modelo.fc.in_features

    #reemplazamos la capa 'fc' por una nueva capa lineal adaptada exactamente a tus clases (ej. num_classes=2)
    #esta nueva capa si tendra 'requires_grad=True' (aprendera los patrones nuevos)
    modelo.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    return modelo

if __name__ == "__main__":
    #Inicializamos el modelo 2 clases personalizadas
    num_clases_proyecto = 2
    modelo_transfer = configurar_modelo_transfer_learning(num_classes=num_clases_proyecto)
    print(f"[EXITO] Modelo ResNet18 cargado y adaptado correctamente.")
    print(f"[INFO] Nueva cabeza de clasificacion configurada para {num_clases_proyecto} clases.")

    #prueba rapida de forward pass con una imagen simulada estandar de la industria: 3 canales, 224x224 pixeles
    tensor_imagen_prueba = torch.randn(2, 3, 224, 224) #Batch de 2 imagenes RGB 224x224
    salida = modelo_transfer(tensor_imagen_prueba)

    print(f"[INFO] Forma del tensor de entrada: {tensor_imagen_prueba.shape}")
    print(f"[INFO] Forma del tensor de salida (Logits): {salida.shape}")

