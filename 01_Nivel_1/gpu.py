import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class AlmacenDePacientes(Dataset):
    def __init__(self):
        self.entradas = torch.randn(100, 1)  
        self.metas = self.entradas * 3.0 + 5.0

    def __len__(self):
        return len(self.entradas)

    def __getitem__(self, indice):
        paciente_x = self.entradas[indice]
        diagnostico_y = self.metas[indice]
        return paciente_x, diagnostico_y

almacen = AlmacenDePacientes()
mesero = DataLoader(almacen, batch_size=20, shuffle=True)

#le decimos que si no esta disponible cuda para pytorch tome la cpu
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware Seleccionado: {dispositivo}")

class NeuronaMedica(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1, bias=True)
    def forward(self, x):
        return self.fc(x)

modelo = NeuronaMedica()
#le decimos que dentro del modelo ejecute esa variable dispositivo: con la palabra reservada .to
modelo = modelo.to(dispositivo)
optimizador = torch.optim.SGD(modelo.parameters(), lr=0.21)

print("--- INICIANDO ENTRENAMIENTO CON MOTOR DE DATOS ---")

for epoca in range(3):  
    print(f"\n🎬 Iniciando Época {epoca+1}")
    
    for num_viaje, (bandeja_entradas, bandeja_metas) in enumerate(mesero):
        #pasamos por cada epoca una cantidad de datos
        bandeja_entradas =  bandeja_entradas.to(dispositivo)
        bandeja_metas = bandeja_metas.to(dispositivo)
        #seguimos con el protocolo
        predicciones = modelo(bandeja_entradas)
        perdida = torch.mean((predicciones - bandeja_metas) ** 2)
        perdida.backward()
        optimizador.step()  
        optimizador.zero_grad()
        
        print(f"   ↳ Viaje del mesero #{num_viaje+1} | Trajo {len(bandeja_entradas)} pacientes | Pérdida en este lote: {perdida.item():.4f}")

print("\n--- ¡ENTRENAMIENTO COMPLETADO! ---")
for nombre, tensor_interno in modelo.named_parameters():
    print(f"-> {nombre} aprendido por la IA: {tensor_interno.data.item():.4f}")

# =====================================================================
# ⚡ TRUCOS SENIOR: GESTIÓN DE HARDWARE Y MEMORIA VRAM (NIVEL 3)
# =====================================================================

# 🖥️ TIP 1: CÓDIGO HÍBRIDO (AGNOSTIC DEVICE)
# Siempre usa 'torch.device("cuda" if torch.cuda.is_available() else "cpu")'.
# Esto hace que tu código funcione en supercomputadoras de Google o en laptops viejas sin romperse.

# 📦 TIP 2: ¿POR QUÉ MUDAR LOS DATOS ADENTRO DEL BUCLE?
# Si mudas todos tus millones de datos a la GPU al principio del script, la memoria VRAM
# de la tarjeta gráfica se llenará de inmediato (Out of Memory Error) y la PC colapsará.
# Al usar '.to(device)' ADENTRO del bucle, solo subes 20 datos a la vez. Cuando el optimizador
# termina, esos 20 datos se borran y entra la siguiente bandeja, manteniendo la GPU limpia.

# 📍 TIP 3: REVISAR LA UBICACIÓN FÍSICA
# Si alguna vez tienes dudas de en qué chip está viviendo un tensor, puedes investigarlo
# usando el atributo '.device'. Por ejemplo: 'print(predicciones.device)'.

