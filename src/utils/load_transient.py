#!/usr/bin/env python3
"""
inspect_transient_simple.py

Este script carga un único archivo .npy de transiente (formato H×W×T×C)
y dibuja:
  1) La instantánea temporal t en el canal R.
  2) La respuesta temporal del píxel (i, j) en el canal R.

Uso:
    python inspect_transient_simple.py \
        --npy_path /ruta/a/scene_000_transient.npy \
        --t 90 \
        --pixel 0 0
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_transient(npy_path: str) -> np.ndarray:
    """
    Carga un .npy de transitorio y verifica que tenga 4 dimensiones (H, W, T, C).
    """
    data = np.load(npy_path)
    if data.ndim != 4:
        raise ValueError(f"{npy_path} no es un volumen 4D; tiene shape {data.shape}")
    return data.astype(np.float32)

def plot_time_instant(data_transient: np.ndarray, t: int):
    """
    Muestra la instantánea temporal t (heatmap) del canal R.
    data_transient: shape (H, W, T, 3)
    """
    H, W, T, C = data_transient.shape
    if not (0 <= t < T):
        raise ValueError(f"t={t} fuera de rango [0, {T-1}]")
    # Extraemos canal R en la rebanada temporal t
    img = data_transient[:, :, t, 0]
    # Ajustamos orientación: first transpose, then flip left-right
    img_plot = np.fliplr(img.T)
    plt.figure(figsize=(4, 4))
    plt.imshow(img_plot, cmap='hot')
    plt.colorbar(label="Radiancia (canal R)")
    plt.axis('off')
    plt.xlabel("Relay wall X")
    plt.ylabel("Relay wall Y")
    plt.title(f"t_idx = {t}")
    plt.show()

def pixel_time_response(data_transient: np.ndarray, i: int, j: int):
    """
    Grafica la respuesta temporal del píxel (i, j), canal R.
    data_transient: shape (H, W, T, 3)
    """
    H, W, T, C = data_transient.shape
    if not (0 <= i < H and 0 <= j < W):
        raise ValueError(f"({i}, {j}) fuera de rango H×W = ({H}, {W})")
    curve = data_transient[i, j, :, 0]  # canal R
    plt.figure(figsize=(5, 3))
    plt.plot(curve)
    plt.xlabel("Time index")
    plt.ylabel(f"Captured radiance at pixel ({i}, {j})")
    plt.title("Respuesta temporal")
    plt.show()

def main(npy_path: str, t: int, i: int, j: int):
    data = load_transient(npy_path)
    print(f"Cargado: {npy_path}  |  shape = {data.shape}")

    # Plot instantánea t
    plot_time_instant(data, t)

    # Plot respuesta temporal de píxel (i, j)
    pixel_time_response(data, i, j)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspecciona un .npy de volumen transitorio y muestra instantánea y curva temporal."
    )
    parser.add_argument(
        '--npy_path',
        type=str,
        required=True,
        help='Ruta al archivo .npy que contiene el volumen transitorio (H×W×T×3)'
    )
    parser.add_argument(
        '--t',
        type=int,
        default=90,
        help='Índice temporal a visualizar (0 ≤ t < T). Por defecto 90'
    )
    parser.add_argument(
        '--pixel',
        type=int,
        nargs=2,
        metavar=('i', 'j'),
        default=[0, 0],
        help='Coordenadas del píxel (i j) para graficar la respuesta temporal. Por defecto 0 0'
    )
    args = parser.parse_args()
    main(args.npy_path, args.t, args.pixel[0], args.pixel[1])

