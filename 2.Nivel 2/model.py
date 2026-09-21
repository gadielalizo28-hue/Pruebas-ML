import torch
import torch.nn as nn

#este archivo es el modelo que vamos a modularizar y previamente en otro archivo heredar y hacemos un control

class RedProfesional(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        #la banda transportadora profesional: nn.Sequential
        self.red_completa = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Los datos fluyen directamente por el bloque secuencial
        return self.red_completa(x)

