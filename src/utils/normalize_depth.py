import os
import re
import numpy as np
from pathlib import Path

def load_pfm(file_path: str) -> np.ndarray:
    with open(file_path, 'rb') as f:
        header = f.readline().decode('latin-1').rstrip()
        if header != 'Pf':
            raise Exception(f"No es un PFM válido (se esperaba 'Pf', se encontró '{header}')")

        dim_line = f.readline().decode('latin-1').strip()
        m = re.match(r'^(\d+)\s+(\d+)\s*$', dim_line)
        if not m:
            raise Exception(f"Formato de dimensiones incorrecto en PFM: '{dim_line}'")
        width, height = map(int, m.groups())

        scale_line = f.readline().decode('latin-1').strip()
        scale = float(scale_line)
        endian = '<' if scale < 0 else '>'
        if scale < 0:
            scale = -scale

        data = np.fromfile(f, endian + 'f')
        expected = width * height
        if data.size != expected:
            raise Exception(f"PFM inesperado: {data.size} elementos, esperados {expected}")

        data = data.reshape((height, width))
        data = np.flipud(data)
        return data.astype(np.float32)


def save_pfm(image: np.ndarray, filename: str, scale: float = 1.0) -> None:
    H, W = image.shape
    with open(filename, 'wb') as f:
        f.write(b'Pf\n')
        f.write(f"{W} {H}\n".encode('utf-8'))

        if image.dtype.byteorder == '<' or (image.dtype.byteorder == '=' and np.little_endian):
            scale = -scale
        f.write(f"{scale}\n".encode('utf-8'))

        data_to_write = np.flipud(image.astype(np.float32))
        data_to_write.tofile(f)


def normalize_depth_to_minus1_plus1(depth: np.ndarray,
                                    d_min: float,
                                    d_max: float) -> np.ndarray:
 
    depth_clamped = np.clip(depth, d_min, d_max).astype(np.float32)
    midpoint = 0.5 * (d_min + d_max)
    scale = 2.0 / (d_max - d_min)
    depth_normalized = (depth_clamped - midpoint) * scale
    return depth_normalized


def process_all_pfms(root_in: str,
                     root_out: str,
                     d_min: float,
                     d_max: float) -> None:
    root_in = Path(root_in)
    root_out = Path(root_out)
    for pfm_path in root_in.rglob("*.pfm"):
        relative_path = pfm_path.relative_to(root_in)
        stem = relative_path.stem  
        new_name = f"{stem}_norm.pfm"
        destino = root_out / relative_path.parent / new_name

        destino.parent.mkdir(parents=True, exist_ok=True)

        depth_raw = load_pfm(str(pfm_path))
        depth_norm = normalize_depth_to_minus1_plus1(depth_raw, d_min, d_max)
        save_pfm(depth_norm, str(destino), scale=1.0)
        print(f"Normalizado: '{pfm_path}' → '{destino}' (shape {depth_norm.shape})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normaliza todos los depth‐maps .pfm bajo una carpeta.")
    parser.add_argument(
        '--input_root', type=str, required=True,
        help="Directorio raíz que contiene subcarpetas con archivos '.pfm' de depth."
    )
    parser.add_argument(
        '--output_root', type=str, required=True,
        help="Directorio donde se guardarán los depth normalized (.pfm)."
    )
    parser.add_argument(
        '--d_min', type=float, required=True,
        help="Profundidad mínima real."
    )
    parser.add_argument(
        '--d_max', type=float, required=True,
        help="Profundidad máxima real."
    )
    args = parser.parse_args()

    process_all_pfms(
        root_in = args.input_root,
        root_out = args.output_root,
        d_min   = args.d_min,
        d_max   = args.d_max
    )


## d_min   = 0.0,
## d_max   = 1.652650


