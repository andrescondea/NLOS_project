# src/utils/losses.py

import torch.nn as nn

def get_loss(loss_name: str):
    """
    Devuelve la función de pérdida que indique 'mse' o 'l1'.
    El paper final optó por MSE, ya que L1 mostró inestabilidades.
    """
    if loss_name.lower() == 'mse':
        return nn.MSELoss()
    elif loss_name.lower() == 'l1':
        return nn.L1Loss()
    else:
        raise ValueError(f"Loss '{loss_name}' no reconocida.")
