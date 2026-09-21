import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

class DatasetMedicoReal(Dataset):
    """
    Dataset profesional para cargar datos médicos reales optimizados en memoria.
    """
    def __init__(self) -> None:
        # 1. Cargamos datos reales del mundo real (Scikit-Learn Dataset)
        data = load_breast_cancer()
        X, y = data.data, data.target 
        #x mayuscula contiene las caracteristicas, y minuscula contiene las etiquetas de clasificación (0 o 1)
        #data.data es un array de numpy con las características de cada muestra 
        #data.target es un array de numpy con las etiquetas de cada muestra

        # 2. Truco Senior: Normalizar los datos reales (Media 0, Desviación 1)
        # Las redes neuronales odian los números desbalanceados; esto acelera el aprendizaje un 100%.
        scaler = StandardScaler()
        X_normalizado = scaler.fit_transform(X)
        #aqui aplicamos el scaler a las caracteristicas para que tengan media 0 y desviacion estandar 1, esto es importante para que la red neuronal aprenda mejor y mas rapido

        # 3. Convertimos a tensores de PyTorch una sola vez (Atajo de velocidad)
        self.X_tensor = torch.tensor(X_normalizado, dtype=torch.float32)
        self.y_tensor = torch.tensor(y, dtype=torch.long) # Etiquetas de clasificación

    def __len__(self) -> int:
        # Retorna el total de muestras reales en el dataset
        return len(self.X_tensor)

    def __getitem__(self, idx: int):
        # Retorna un solo par de datos y su etiqueta real correspondiente al índice
        return self.X_tensor[idx], self.y_tensor[idx]
        #idx es el índice de la muestra que queremos obtener, y retornamos un par (características, etiqueta) correspondiente a ese índice