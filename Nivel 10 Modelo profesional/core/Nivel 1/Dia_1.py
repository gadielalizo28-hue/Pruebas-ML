import torch
import numpy as np

# Scalar
scalar = torch.tensor(7)
print(scalar)

#Para crear un tensor con datos preexistentes, utilice torch.tensor().

#Para crear un tensor con un tamaño específico, utilice torch.*las operaciones de creación de tensores (consulte Operaciones de creación ).

#Para crear un tensor del mismo tamaño (y tipos similares) que otro tensor, utilice torch.*_likelas operaciones de creación de tensores (consulte Operaciones de creación ).

#Para crear un tensor del mismo tipo pero de tamaño diferente a otro tensor, utilice tensor.new_*operaciones de creación.

#Existe un constructor heredado torch.Tensorcuyo uso no se recomienda. Utilice torch.tensor()en su lugar.

#crear un tensor a partir de una lista de python:
scalar = torch.tensor([42])
vector = torch.tensor([1.5, 2.3, 3.1])
matrix = torch.tensor([[1, 2], [3, 4]])

print(f"Tensor 2D (matriz):\n{matrix}")
print(f"Forma (shape): {matrix.shape}")
print(f"Tipo de datos (dtype): {matrix.dtype}")

#tensores inicializados automaticamente (vitales para pesos de redes neuronales)
zeros = torch.zeros((2, 3)) #matriz de 2x3 llena de ceros
ones = torch.ones((2, 3)) #matriz de 2x3 llena de unos
random_tensor = torch.rand((2, 3)) #valores aleatorios entre 0 y 1 (distribucion uniforme)

#operaciones elemento a elemento(element-wise): su sumas dos tensores con la misma forma, las operaciones ocurren
#posicion por posicion:
a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20, 30]) 

print(a + b) #resultado: tensor([11, 22, 33])
print(a * b) #resultado: tensor([10, 40, 90])

#multiplicacion de matrices (dot product / Matrix de multiplicacion): para conectar capas de redes neuronales, multiplicamos
#filas por columnas. En pytorch, usamos el operador @ o la funcion torch.matmul()
tensor_a = torch.tensor([[1, 3],
                        [2, 4]])

tensor_b = torch.tensor([[5, 6],
                        [7, 8]])
#multiplicamos 1x5 + 3x7, luego 1x6 + 3x8, luego 2x5 + 4x7, luego 2x6 + 4x8

#multiplicacion matricial tradicional:
resultado = tensor_a @ tensor_b
print(resultado)

#regla de oro: para multiplicar matrices, el numero de columnas de la primera matriz debe ser igual al numero de filas de la segunda matriz.

#reshape o view: cambian las dimensiones de un tensor sin alterar los datos internos, siempre y cuando el numero total
#de elementos sea el mismo. Por ejemplo, podemos cambiar un tensor de 2x3 a 3x2:

#un vector de 12 elementos
x = torch.arange(12)
print("Original:", x.shape)

#lo convertimos en una matriz de 3x4
x_reshaped = x.view(3, 4)
print("Reshaped (3x4): \n", x_reshaped)

#0 en un tensor tridimensional: podemos cambiar un tensor de 2x3x4 a 4x3x2 usando view o reshape, siempre y 
#el numero total de elementos sea el mismo.
x_3d = x.reshape(2, 2, 3)
print("Reshaped 3D:\n", x_3d) 
#esto sirve para cambiar la forma de los datos, por ejemplo, cuando se trabaja con imágenes o secuencias en redes neuronales.

#permute o transpose: reordenan los ejes del tensor. Muy usado en vision por computador cuando pasas de formato a 
#canales al final (Alto, Ancho, Canales) a formato de pytorch al inicio (Canales, Alto, Ancho). Por ejemplo, 
# si tenemos un tensor de forma (2, 3, 4) y queremos cambiarlo a (4, 2, 3), podemos usar permute:

#reshape o view solo cambian la forma del tensor, mientras que permute o transpose reordenan los ejes del tensor.
#Por ejemplo, si tenemos un tensor de forma (2, 3, 4) y queremos cambiarlo a (4, 2, 3), podemos usar permute

#inmutabilidad logica: Aunque puedas cambiar los valores detro de un tensor, cambiar su estructura (Shape) requiere
#crear vistas o resignaciones en memoria. Por ejemplo, si tienes un tensor de forma (2, 3) y quieres cambiarlo a (3, 2), no puedes simplemente asignar
#una nueva forma, sino que debes crear un nuevo tensor con la forma deseada usando view o reshape. Esto es importante
#porque garantiza que los datos subyacentes no se modifiquen accidentalmente y permite optimzaciones

#torch.randn((shape)) : crea numero aleatorios entre 0 y 1 (usado para pesos iniciales)

#torch.zeros_like(tensor): crea un tensor de ceros con exactamente la misma forma que otro tensor dado. Esto es útil 
#cuando quieres inicializar un tensor de ceros que coincida con la forma de otro tensor existente.

#tensor.shape: devuelve la forma de el tensor

#tensor.dtype: devuelve el tipo de datos del tensor

#tensor.device: devuelve el dispositivo en el que se encuentra el tensor (CPU o GPU)

cubo = torch.zeros((2, 3, 4)) #tensor de ceros de 3 dimensiones
print(f"Cubo shape (Alto, Ancho, Profundidad): {cubo.shape}\n")

#creaciones especiales usadas en produccion:

aleatoios = torch.rand((3, 3)) #tensor de 3x3 con valores aleatorios
print(f"Tensor aleatorio 3x3:\n{aleatoios}\n")

#crear un tensor identico a otro tensor pero lleno de 0:
tensor_cero = torch.zeros_like(aleatoios)
print(f"Tensor de ceros con la misma forma que el tensor aleatorio:\n{tensor_cero}\n")

#inspeccion de metadaos criticos de un tensor:
mi_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"Tipo de datos (dtype - estandar ML float32): {mi_tensor.dtype}")

print(f"Dispositivo en el que se encuentra el tensor: {mi_tensor.device}")

#truco senior: mover un tensor a la GPU si esta disponible:
if torch.cuda.is_available():
    tensor_en_gpu = mi_tensor.to('cuda')
    print(f"Tensor en GPU: {tensor_en_gpu.device}")

else:
    print("GPU no disponible, usando CPU.")

#para no usar datos aleatorios:

#torch.tensor(tu_lista, dtype=torch.float32) #convierte listas a matrices de corchetes en un tensor listo para entrenar

#torch.from_numpy(mi_array_numpy) #convierte un array de numpy directamente a tensor sin copiar los datos de memoria

#tensor.numy: Pasa un tensor de pytorch a numpy (muy util para graficar con matplotlib)

#type(torch.floa32): cambia la precision numerica de tus datos introducidos:

print("\n--- 0. CREACION DE TENSORES ---\n")

# CASO 1: Tus propios datos escritos a mano (Listas anidadas de Python)
# Imagina que son 3 muestras, y cada una tiene 2 características (Horas de estudio, Horas de sueño)
mis_datos_propios = [
    [5.0, 8.0],
    [2.0, 5.0],
    [8.0, 7.0]
]

# Los convertimos en tensor y FORZAMOS que sean float32 (vital para redes neuronales)
tensor_propios = torch.tensor(mis_datos_propios, dtype=torch.float32)

print("--- 1. DESDE LISTA DE PYTHON ---")
print(tensor_propios)
print("Forma (Shape):", tensor_propios.shape)
print("Tipo de dato (Dtype):", tensor_propios.dtype)


# CASO 2: Tus propios datos usando NumPy (Lo que harás al leer un archivo CSV)
# Supongamos que tienes un array de NumPy con tus datos del mundo real
array_de_numpy = np.array([[10, 20], [30, 40]])

# Lo transformamos a tensor de PyTorch profesionalmente
tensor_desde_numpy = torch.from_numpy(array_de_numpy).type(torch.float32)

print("\n--- 2. DESDE NUMPY A PYTORCH ---")
print(tensor_desde_numpy)


# CASO 3: El viaje de vuelta (De Tensor a NumPy para graficar o exportar)
# Si en algún momento necesitas sacar tus números de PyTorch para mostrarlos con gráficos:
matriz_regreso_a_numpy = tensor_propios.numpy()

print("\n--- 3. DE REGRESO A NUMPY ---")
print(matriz_regreso_a_numpy)
print(type(matriz_regreso_a_numpy)) # Verás que vuelve a ser un ndarray de NumPy

#Ejercicio:

sensor_data = np.array([1, 2, 3, 4, 5, 6, 7, 8 ,9 ,10 ,11, 12])

#lo convertimos a pytorch sin copiar datos:
sensor_tensor = torch.from_numpy(sensor_data).type(torch.float32) #siempre cambia el tipo a float32

#cambiamos la forma con un reshape(3x4):
reshaped_sensor_tensor = sensor_tensor.reshape(3, 4)
print("Forma (Shape):", reshaped_sensor_tensor.shape)

#simulacion de mascara(creacion automatica):

pesos = torch.ones((3, 4))

#multiplicacion elemento a elemento con la mascara de pesos:
masked_tensor = reshaped_sensor_tensor * pesos

#lo convertimos en un array de numpy para poder enviarlo al software de graficacion:
masked_numpy_array = masked_tensor.numpy()

#Ejercicio 2:

nuevo_array = np.array([1, 2, 3, 4, 5, 6, 7, 8])

sensor_nuevo = torch.from_numpy(nuevo_array).type(torch.float32)

sensor_nuevo_reshaped = sensor_nuevo.reshape(4, 2)

array = torch.zeros_like(sensor_nuevo_reshaped)

print(f"Dispositivo en el que se encuentra el tensor: {array.device}")

#ejecicio 3:

array_nuevo = np.array([10, 20, 30])

nuevo_tensor = torch.from_numpy(array_nuevo).type(torch.float32)

ultimo_tensor = torch.ones_like(nuevo_tensor)

print("Ejercicio 3\n")
print(ultimo_tensor)



