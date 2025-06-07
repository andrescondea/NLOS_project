# src/utils/metrics.py

import torch
import torch.nn.functional as F

def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calcula Root-Mean-Square Error (RMSE) entre pred y target,
    considerando sólo los píxeles que se clasifican como 'foreground'
    en el ground-truth.
    Ambas tensores tienen shape (B, H, W).
    """
    # Primero, definimos un umbral que considere “foreground” en target.
    # En el paper, ellos normalizan profundidad a [-1,1] y dicen que z <= 1 y z > -1 es foreground.
    # Aquí: si target > -1 (es decir, no fondo), lo consideramos foreground.
    # Ajusta el umbral si tus depth maps están en [0,1].
    thresh = -0.9999  # cualquier valor > -1
    mask = (target > thresh).float()           # (B, H, W), 1 donde hay foreground, 0 en background

    diff2 = (pred - target) ** 2                # (B, H, W)
    num_fg = mask.sum(dim=[1,2])                # (B,) número de pixeles foreground por ejemplo

    # evitar división por cero:
    num_fg = torch.clamp(num_fg, min=1.0)

    sum_fg = (diff2 * mask).sum(dim=[1,2])      # suma de squared error sólo en foreground
    mse_fg = sum_fg / num_fg                    # (B,)
    rmse_fg = torch.sqrt(mse_fg)                # (B,)
    return rmse_fg.mean()                       # promedio sobre el batch

def accuracy_fg_bg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calcula accuracy de clasificación foreground vs background.
    - Clasificamos como foreground si depth > thresh_fg, 
      como background si depth <= thresh_fg.
    - El paper utiliza threshold en z normalizado: z <= -1 es background.  
      Nosotros asumiremos que target está en [-1,1]. 
      Por tanto: 
         - background si pred <= -1
         - foreground si pred > -1
    """
    thresh = -0.9999
    # Etiquetas del ground-truth:
    gt_fg = (target > thresh).float()    # 1 donde hay objeto real, 0 en background
    # Predicciones convertidas a binario:
    pred_fg = (pred > thresh).float()    # 1 donde model predicts foreground, 0 en background

    correct = (pred_fg == gt_fg).float() # (B, H, W)
    acc  = correct.mean()                # media sobre (B*H*W)
    return acc

