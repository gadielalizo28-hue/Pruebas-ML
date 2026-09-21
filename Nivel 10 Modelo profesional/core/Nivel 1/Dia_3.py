import torch
import numpy as np 

#el autograd y el grafo computacional:

#el grafo computacinal: cada vez que haces una operacion con un tensor, la libreria construye silenciosamente un mapa 
#mental invisible de nodos conectados(un grafo).

#el Autograd(Diferenciacion automatica): Cuando aplicas la funcion .backward() al final de una prediccion, Pytorch rcorre
#ese mapa en sentido inverso aplicando calculo multivariado. Con esto, le dice exactamente a cada numero cuanto debe
#cambiar para cometer menos errores la proxima vez.

#Codigos:

#Activar rastreo de gradiantes: requires_grad=True : le avisa a PyTorch que guarde memoria para calcular derivadas de ese tensor

#Calcular derivadas: tensor.backward() : Ejecuta el motor de autograd recorriendo el grafo hacia atras

#Ver el gradiante calculado: tensor.grad : Muestra cuanto debe corregirse numericamente ese peso

#Apagar el motor de calculo: with torch.no_grad(): Contesto vital para validacion/evaluacion; ahorra memoria masivamente

#desconectar el grafo: tensor.detach() : Saca un tensor del sistema de rastreo para pasarlo limpio a numpy o graficos

#EJEMPLO:

# 1. Definimos un peso inicial que requiere cálculo de gradientes
peso = torch.tensor([2.0], requires_grad=True)
entrada = torch.tensor([3.0])

# 2. Operación matemática (Simulando una capa muy simple: y = w * x)
salida = peso * entrada

# 3. Suponemos que la pérdida (el error) es el valor de la salida
perdida = salida

# 4. Disparamos el motor de Autograd hacia atrás
perdida.backward()

# 5. Vemos cuánto afectó el peso al resultado final
print("El gradiente (derivada) del peso es:", peso.grad)

#EJEMPLO 2: con tus propios numeros

# TUS DATOS: Factor de calibración inicial (requiere gradientes para aprender)
factor_calibracion = torch.tensor([1.5], requires_grad=True)

# TUS DATOS: Lectura real de flujo de combustible detectada por el sensor
flujo_sensor = torch.tensor([4.2])

# 1. Proyección lineal del sistema
resultado_modelo = factor_calibracion * flujo_sensor ##aqui se multiplica 1.5 x 4.2 = 6.3 y el objetivo era 8
print("--- TUS DATOS EN EL MODELO ---")
print("Resultado estimado del modelo:", resultado_modelo.item())

# 2. Supongamos que calculamos el error y queremos aprender de él
# Forzamos una pérdida basada en una meta ideal de 8.0
error = (resultado_modelo - 8.0) ** 2 #la meta era 8 asi que calculamos a la potencia de 2 el resultado menos la meta
print("Pérdida (Error calculado):", error.item())

# 3. BACKWARD: Calculamos la derivada para saber cómo ajustar el factor
error.backward() 
#esto lo que hace es una operacion distinta, multiplica x2 el resultado menos la meta y el resultado de eso lo 
#multiplica x4.2 es decir por la derivada interna.
print("Gradiente calculado (Dirección y fuerza de corrección):", factor_calibracion.grad) #dando asi -14.28

# 4. ESCENARIO SENIOR (Ahorro de memoria):
# Cuando pasas a fase de pruebas y ya no quieres gastar recursos calculando derivadas:
print("\n--- MODO DE EVALUACIÓN SEGURA ---")
with torch.no_grad():
    prueba_segura = factor_calibracion * 5.0
    print("Prueba ejecutada sin consumir memoria de gradientes:", prueba_segura.item())

#la maquina de por si asi los calculos, lo que tu debes hacer es estructurar bien esto y elegir la funcion de perdida
#correcta, la funcion de perdida cambia segun nuestros objetivos, a continuacion te dare ejemplos:

#Error Absoluto Medio/ MAE (Mean absolute error): (resultaod - meta) ** 2: para problemas de regresion, 
#(cuando quieres predecir un numero continuo, como el precio de una casa, la temperatura exacta o el valor de una accion).
#AL elevar el cuadrado, castigas mas fuerte los errores grandes

#tambien hay otra version de esa formula y es: resultaod - meta (se usa un valor absoluto en vez de elevar al cuadrado)
#Se usa tambien para regresion pero cuando tienes datos con muchos valores atipicos(outliers), ya que el cuadrado del MSE
#exagera demasiado los errores grandes y desbalancea el entrenamiento.

#Entropia Cruzada Binaria (binary Cross-Entropy): Se usa cuando la respuesta es si o no, como 0 o 1, spam o no spam,
#perro o gato. Aqui no mides distancias numericas si no probabilidades (el modelo dice 95% de probabilidad de que sea un gato entonces el modelo decide)

import torch.nn as nn

# 1. Lo que predice tu modelo (probabilidades en bruto, ej: 0.85 de probabilidad de que sea clase 1)
prediccion = torch.tensor([0.85], dtype=torch.float32, requires_grad=True)

# 2. La etiqueta real (0.0 o 1.0)
etiqueta_real = torch.tensor([1.0], dtype=torch.float32)

# 3. Llamamos a la función de pérdida de Entropía Cruzada Binaria
criterio_bce = nn.BCELoss() #codigo de entropia
perdida = criterio_bce(prediccion, etiqueta_real) #la variable con el codigo entropia sobre las dos variables anteriores

print("Pérdida por Entropía Cruzada Binaria:", perdida.item())

# 4. ¡La estructura no cambia! El autograd funciona idéntico:
perdida.backward()
print("Gradiente calculado:", prediccion.grad) #Muestra cuanto debe corregirse numericamente ese peso

#Entropia Cruzada Categoria(Categorial Croos-Entropy): Se usa para clasificacion multiclase(cuando tienes mas de dos opciones
#como reconocer digitos del 0 al 9 o clasificar si una foto es un carro, una moto, un avion, o un barco)

# 1. Supongamos que tu modelo predice puntuaciones (logits) para 3 clases (ej: Clase 0, Clase 1, Clase 2)
predicciones_clases = torch.tensor([[2.5, 0.1, -1.2]], dtype=torch.float32, requires_grad=True)

# 2. La clase real correcta (aquí indicamos que la respuesta verdadera es la Clase '0')
clase_verdadera = torch.tensor([0], dtype=torch.long)

# 3. Llamamos a la función de pérdida categórica
criterio_multiclase = nn.CrossEntropyLoss() #funcion de perdida categoria
perdida = criterio_multiclase(predicciones_clases, clase_verdadera) #luego la variable que tiene dentro la funcion sobre
#las dos variables anteriores

print("Pérdida por Entropía Cruzada Categórica:", perdida.item())

# 4. Y de nuevo... ¡Misma estructura! El backward opera sin modificar nada del flujo:
perdida.backward()
print("Gradientes calculados para las 3 clases:\n", predicciones_clases.grad) #Muestra cuanto debe corregirse numericamente ese peso


#EJERCICIO 1:

#Tensor con valor incial
print("\nEjercicio 1:\n")
peso_sistema = torch.tensor([3.0], requires_grad=True)

#Tensor de lectura
lectura_tensor = torch.tensor([2.0])

#calculamos la prediccion del modelo
prediccion_modelo = peso_sistema * lectura_tensor
print("Prediccion de el modelo:\n", prediccion_modelo.item())

#calculamos el error
perdida_a = (prediccion_modelo - 10.0) ** 2
print("Perdida por Entropia Cruzada Binaria:\n", perdida_a.item())

#Backward Pass y gradiantes:
perdida_a.backward()
print("Gradiante Backward:\n", peso_sistema.grad)

#Modo seguro:
with torch.no_grad():
    modo_seguro = peso_sistema * 4.0
    print("Modo Seguro:\n", modo_seguro.item())

#Ejercicio 2:

print("\nEjercicio 2:/\n")
factor_aceleracion = torch.tensor([1.5], requires_grad=True)
tiempo_impulso = torch.tensor([4.0])

#calculamos:
velocidad_final = factor_aceleracion * tiempo_impulso
perdida_b = (velocidad_final - 10.0) ** 2
print("La perdida sobre la Velocidad Final:\n", perdida_b.item())

#backward y gradientes:
perdida_b.backward()
print("Pedida con backward:\n", factor_aceleracion.grad)

#modo seguro:
with torch.no_grad():
    modo_seguro_b = factor_aceleracion * 2.0
    print("Modo seguro:\n", modo_seguro_b.item())


#Ejercicio 3:

print("Ejercicio 3:\n")
sensores = torch.tensor([
    [10.0, 30.4, 20.3],
    [30.5, 53.2, 32.4],
    [22.3, 27.4, 17.3],
    [34.3, 37.5, 29.9]
], dtype=torch.float32, requires_grad=True)

print("Esta es su forma:", sensores.shape)
print("Este es su tipo:", sensores.dtype)

extraccion_datos = sensores[1:3, :]
print("\nExtraccion de Datos:\n", extraccion_datos)

reagrupado = sensores.reshape(2, 6)
print("\nNueva Forma del Tensor:\n", reagrupado)

calculo_1 = sensores * 2.0
perdida_cal = (calculo_1 - 50.0) ** 2
perdida_total = perdida_cal.sum()
print("\nSuma total de la pérdida:", perdida_total.item())

perdida_total.backward()
print("\nDiferenciacon Automatica:", sensores.grad)

with torch.no_grad():
    prueba = sensores * 1.1
    print("\nPrueba Segura:", prueba)








