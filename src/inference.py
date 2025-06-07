#!/usr/bin/env python3
"""
infer.py

Inferencia de DepthMap NLoS:
  - Carga un checkpoint PyTorch (.pth)
  - Lee un archivo .npy con la respuesta transitoria (32×32×256[,3])
  - Invoca el modelo DepthMap
  - Guarda el depth map resultante (64×64) en formato PFM
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from models.depthmap import DepthMap

def write_pfm(filename: str, image: np.ndarray, scale: float = -1.0):
    """
    Escribe un array 2D float32 en formato PFM (escala negativa = little endian).
    """
    if image.ndim != 2:
        raise ValueError("PFM only supports 2D grayscale images.")
    height, width = image.shape
    with open(filename, 'wb') as f:
        # Cabecera
        header = f"Pf\n{width} {height}\n{scale}\n"
        f.write(header.encode('ascii'))
        # PFM almacena las filas de abajo hacia arriba
        flipped = np.flipud(image)
        # Escribir datos como floats little endian
        flipped.astype('<f4').tofile(f)

def preprocess_transient(path: str) -> torch.Tensor:
    """
    Carga un .npy transiente y devuelve un tensor (1,1,256,32,32).
    """
    arr = np.load(path).astype(np.float32)
    # si viene con canal extra (...,3) tomamos el canal 0
    if arr.ndim == 4:
        arr = arr[..., 0]
    # arr ahora (H=32, W=32, T=256)
    if arr.shape != (32, 32, 256):
        raise RuntimeError(f"Se esperaba (32,32,256), pero vino {arr.shape}")
    # transponer a (T, H, W)
    arr = np.transpose(arr, (2, 0, 1))
    # añadir canal y batch: (1, 1, 256, 32, 32)
    arr = arr[None, None, ...]
    return torch.from_numpy(arr)

def main(checkpoint: str, transient_npy: str, output_pfm: str, device: str):
    # Configurar device
    dev = torch.device(device)
    # Instanciar y cargar modelo
    model = DepthMap().to(dev)
    sd = torch.load(checkpoint, map_location=dev)
    # si guardaste state_dict puro:
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        model.load_state_dict(sd['model_state_dict'])
    else:
        model.load_state_dict(sd)
    model.eval()

    # Preprocesar input
    x = preprocess_transient(transient_npy).to(dev)

    # Inferir
    with torch.no_grad():
        out = model(x)            # (1,64,64)
    depth = out[0].cpu().numpy()

    # Guardar en PFM
    write_pfm(output_pfm, depth)
    print(f"Depth map guardado en {output_pfm}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inferencia DepthMap NLoS")
    p.add_argument("--checkpoint", required=True,
                   help="Ruta al modelo (.pth)")
    p.add_argument("--input", required=True,
                   help="Numpy .npy con el transient (32×32×256)")
    p.add_argument("--output", required=True,
                   help="Ruta de salida .pfm")
    p.add_argument("--device", default="cuda",
                   help="Device para PyTorch (cuda o cpu)")
    args = p.parse_args()
    main(args.checkpoint, args.input, args.output, args.device)
