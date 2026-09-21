import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""
#definimos una clase con el nombre y entre parentesis el codigo nn.Module que siempre esta
class MiPrimerModeloSenior(nn.Module):

    def __init__(self):
        super().__init__() #el boton de encendido, siempre debe estar
        
        #nn.linear crea 1 peso y 1 sesgo automaticamente y los guardamos en la ranura self.capa_lineal
        self.capa_lineal = nn.Linear(in_features=1, out_features=1, bias=True)

    def forward(self, x):
        #esta funcion hace las matematicas: (x * peso) + sesgo, siempre se usa para pasar los datos por adelante
        return self.capa_lineal(x)

modelo = MiPrimerModeloSenior() #le asignamos una variable al modelo
optimizador = torch.optim.SGD(modelo.parameters(), lr=0.05)
"""
# =====================================================================
# 📚 ENCICLOPEDIA DE TRUCOS Y BUENAS PRÁCTICAS SENIOR (PYTORCH)
# =====================================================================

# 🧠 TRUCO 1: "EL BOTÓN DE PÁNICO" (.shape)
# Si tu código arroja un 'RuntimeError' de dimensiones matemáticas, no adivines.
# Pon un 'print(tensor.shape)' justo en la línea anterior al error. 
# El 99% de los errores en Machine Learning se resuelven mirando las formas (shapes).

# 🚫 TRUCO 2: EL ERROR DEL GRADIENTE ACUMULADO
# PyTorch NO borra los gradientes automáticamente en cada época; los SUMA por defecto.
# Si olvidas poner 'optimizador.zero_grad()', el modelo acumulará los errores pasados,
# los pesos se moverán hacia números gigantescos (infinitos) y el modelo explotará.

# ⚡ TRUCO 3: DETERMINISMO (REPRODUCIBILIDAD)
# Como PyTorch inicializa los pesos de forma aleatoria, cada vez que corras tu script
# los resultados serán ligeramente diferentes. Si quieres que tu código dé SIEMPRE
# exactamente los mismos números para poder estudiar tus experimentos, fija la semilla:
# torch.manual_seed(42)  # El 42 es el número clásico por cultura programadora.

# 🔒 TRUCO 4: DETENER EL APRENDIZAJE ('with torch.no_grad():')
# Cuando vas a PROBAR tu modelo (evaluar si adivina bien), no quieres que calcule
# gradientes ni que gaste memoria RAM calculando derivadas. Envuelve tu código así:
# with torch.no_grad():
#     prediccion_prueba = modelo(entrada_nueva)
# Esto desactiva el motor de derivadas, ahorra un 50% de memoria y acelera el programa.

# 🛠️ TRUCO 5: EL MÉTODO '.item()'
# Nunca uses un tensor de PyTorch directamente dentro de un 'print' de texto normal
# si solo quieres ver el número decimal. Usar '.item()' extrae el número puro de Python
# y libera al recolector de basura de la memoria para que el script vaya más rápido.

#ejemplo en codigo en conjunto:

class MiModelo(nn.Module):
    def __init__(self):
        super(MiModelo, self).__init__()
        self.capa = nn.Linear(in_features=1, out_features=1, bias=True)
        #in_features= son las entradas o los datos
        #out_features= las predicciones de salida
        #bias los sesgos

    def forward(self, x):    
        return self.capa(x)

modelo2 = MiModelo()
entrada = torch.tensor([[3.0]]) #nota: nn.Linear espera un tensor 2D
meta = torch.tensor([[8.0]])
optimizador = torch.optim.SGD(modelo2.parameters(), lr=0.01)

for epoca in range(60):
    prediccion = modelo2(entrada) #pytorch llama automaticamente a forward
    perdida = (prediccion - meta) ** 2

    perdida.backward()
    optimizador.step()
    optimizador.zero_grad()

    print(f"Epoca {epoca+1} | Perdida: {perdida.item():.4f} | Prediccion: {prediccion.item():.4f}")

#cuando la perdida es muy alta disminuye los pasos, asi mismo es al contrario



#Ejemplo en codigo en conjuto pero mas complejo:

print("\nEjercicio 2:\n")
class MiModelo(nn.Module):
    def __init__(self):
        super(MiModelo, self).__init__()
        self.capa1 = nn.Linear(in_features=1, out_features=4, bias=True)
        self.capa2 = nn.Linear(in_features=4, out_features=1, bias=True)
        #in_features= son las entradas o los datos
        #out_features= las predicciones de salida
        #bias los sesgos

    def forward(self, x):    
        x = self.capa1(x)
        x = F.relu(x)
        x = self.capa2(x)
        return x

modelo2 = MiModelo()
entrada = torch.tensor([[3.0]]) #nota: nn.Linear espera un tensor 2D
meta = torch.tensor([[8.0]])
optimizador = torch.optim.SGD(modelo2.parameters(), lr=0.01)

for epoca in range(10):
    prediccion = modelo2(entrada) #pytorch llama automaticamente a forward
    perdida = (prediccion - meta) ** 2

    perdida.backward()
    optimizador.step()
    optimizador.zero_grad()

    print(f"Epoca {epoca+1} | Perdida: {perdida.item():.4f} | Prediccion: {prediccion.item():.4f}")

#cuando la perdida es muy alta disminuye los pasos, asi mismo es al contrario

#podemos agregar un bloque de evaluacion como:
"""
modelo.eval()
with torch.no_grad():
    test.input = torch.tensor([[]])
    resultado = modelo(test.input)
    print(f"\nResultado Final de Prediccion Tras Entrenamiento: {resultado.item():.4f}")
"""



