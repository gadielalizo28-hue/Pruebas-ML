import torch
import torch.nn as nn
# Importamos las herramientas oficiales del motor de datos
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. CONSTRUIMOS EL ALMACÉN (Nuestro Dataset Personalizado)
# =====================================================================
class AlmacenDePacientes(Dataset):
    def __init__(self):
        # Usamos las Factory Functions del Nivel 0 para inventar 100 pacientes
        # Cada paciente tiene 1 dato de entrada (ej: Presión arterial)
        self.entradas = torch.randn(100, 1)  # 100 filas, 1 columna
        
        # Inventamos una regla secreta para la meta: la meta será igual a (entrada * 3) + 5
        self.metas = self.entradas * 3.0 + 5.0

    def __len__(self):
        # Le dice al DataLoader cuántos pacientes tenemos en total en el almacén
        return len(self.entradas)

    def __getitem__(self, indice):
        # El mesero viene aquí y nos pide el paciente número 'indice'
        paciente_x = self.entradas[indice]
        diagnostico_y = self.metas[indice]
        return paciente_x, diagnostico_y


# =====================================================================
# 2. CONFIGURACIÓN INICIAL (Se ejecuta UNA SOLA VEZ al principio)
# =====================================================================
# Instanciamos nuestro almacén con los 100 pacientes
almacen = AlmacenDePacientes()

# Creamos al mesero (DataLoader). 
# Senior Tip: Va a armar bandejas de 20 en 20 pacientes (batch_size=20).
# Como hay 100 pacientes en total, el mesero hará exactamente 5 viajes por época (100 / 20 = 5).
mesero = DataLoader(almacen, batch_size=20, shuffle=True)

# Creamos nuestra neurona clásica
class NeuronaMedica(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1, bias=True)
    def forward(self, x):
        return self.fc(x)

modelo = NeuronaMedica()
optimizador = torch.optim.SGD(modelo.parameters(), lr=0.21)


# =====================================================================
# 3. EL BUCLE DE ENTRENAMIENTO BI-NIVEL (Aquí está el corazón del motor)
# =====================================================================
print("--- INICIANDO ENTRENAMIENTO CON MOTOR DE DATOS ---")

for epoca in range(3):  # Queremos repasar el almacén completo 3 veces
    print(f"\n🎬 Iniciando Época {epoca+1}")
    
    # El segundo bucle 'for' le pide las bandejas al mesero una por una hasta terminar el almacén
    for num_viaje, (bandeja_entradas, bandeja_metas) in enumerate(mesero):
        
        # PASO 1: El Chef procesa los 20 pacientes de la bandeja actual al mismo tiempo
        predicciones = modelo(bandeja_entradas)
        
        # PASO 2: Calculamos el promedio del error de estos 20 pacientes
        perdida = torch.mean((predicciones - bandeja_metas) ** 2)
        
        # PASOS 3, 4 y 5: El ciclo sagrado de actualización
        perdida.backward()
        optimizador.step()  # ¡El optimizador corrige los pesos en cada viaje del mesero!
        optimizador.zero_grad()
        
        print(f"   ↳ Viaje del mesero #{num_viaje+1} | Trajo {len(bandeja_entradas)} pacientes | Pérdida en este lote: {perdida.item():.4f}")

print("\n--- ¡ENTRENAMIENTO COMPLETADO! ---")
for nombre, tensor_interno in modelo.named_parameters():
    print(f"-> {nombre} aprendido por la IA: {tensor_interno.data.item():.4f}")