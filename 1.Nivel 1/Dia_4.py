import torch

#Creacion y manipulacion de tensores:

#Funciones de fabrica: si creas una lista de tensores a mano el cpu gasta tiempo y si tienes muchos datos la pc
#colapsa, en lugar de eso usas las funciones de fabrica:

#torch.zeros(filas, columnas): reserva el espacio y pone los tensores en 0.0. Se usa para inicializar variables
#acumuladoras por sesgos (bias) de una neurona, empezar de 0 es un punto neutral

#torch ones(filas, columnas): reserva el espacio y lo llena con 1.0. Se usa mucho para crear mascaras 

#torch.randn(filas, columnas): reserva el espacio y lo llena con numeros aleatorios (positivos, negativos, como 1.4, 0.2
#2.1) son aleatorios porque las neuronas necesitan empezar con valores diferentes para que cada una aprenda una 
#caracteristica distinta

vector_plano = torch.arange(12)
print(vector_plano.shape) 

matriz = vector_plano.view(3, 4)
print(matriz.shape)

#Un truco Senior del -1:

matriz_auto = vector_plano.view(2, -1)
#i no quieres hacer matematicas, pones -1 y pytorch calcula las dimensiones por ti
print(matriz_auto.shape)

#si tienes un tensor con enteros para cambiarlo a float32:
enteros = torch.tensor([1, 2, 3])
datos_convertidos = enteros.float()

#Ejercicio 1:

print("Ejercicio 1:\n")
tensor = torch.randn([4, 3, 3])

#aqui definimos que el tensor sea un float32
datos_con = tensor.float()
print(tensor)

#cambiamos la forma del tensor
tensor_nuevo2 = tensor.reshape(4, -1)
print("Tensor Forma Nueva:\n",tensor_nuevo2.shape)

#Bucle for para epocas:
print("Bucle para epocas:\n")
peso = torch.tensor([2.0], requires_grad=True) #esto son los tensores que deberian de ir en una clase y variable para manejarlos sin escribirlos uno a uno
sesgo = torch.tensor([0.5], requires_grad=True)

entrada = torch.tensor([3.0])
meta = 10.0 #los dos tensores de siempre

#tamaño de los pasos:
learning_rate = 0.5 

#optimizador:
optimizador= torch.optim.SGD([peso, sesgo], lr=learning_rate)

for epoca in range(5): #aqui hacemos el bucle y definimos el numero de vueltas

    #la prediccion y la operacion matematica
    prediccion = (peso * entrada) + sesgo
    perdida = (prediccion - meta) ** 2
    
    #para que revise sus operaciones de nuevo
    perdida.backward()
    #hace que cada vez los errores sean mas pequeños
    optimizador.step()
    #para limpiar los errores anotados
    optimizador.zero_grad()

    print(f"Epoca {epoca+1} | Perdida: {perdida.item():.4f} | Peso: {peso.item():.4f} | Sesgpo: {sesgo.item():.4f} ")



#ejercicio 2:
print("\nBucle For 2:\n")
peso2 = torch.tensor([2.5], requires_grad=True)
sesgo2 = torch.tensor([0.5], requires_grad=True)

entrada2 = torch.tensor([3.0])
meta2 = 8.0

learning_rate2 = 0.1
optimizador2 = torch.optim.SGD([peso2, sesgo2], lr=learning_rate2)

for epoca in range(10):
    prediccion2 = (peso2 * entrada2) + sesgo2
    perdida2 = (prediccion2 - meta2) ** 2

    perdida2.backward()
    optimizador2.step()
    optimizador2.zero_grad()

    print(f"Epoca {epoca+1} | Perdida: {perdida2.item():.4f} | Peso: {peso2.item():.4f} | Sesgo: {sesgo2.item():.4f} | Prediccion: {prediccion2.item():.4f} ")





