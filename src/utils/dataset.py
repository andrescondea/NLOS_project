# src/data/dataset.py

import re
import numpy as np
import torch
from torch.utils.data import Dataset
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
            raise Exception(f"Esperaba {expected} valores, se leyeron {data.size}")

        data = data.reshape((height, width))
        data = np.flipud(data)
        return data.astype(np.float32)

class NLoSDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir      = Path(root_dir)
        self.depth_root    = self.root_dir / "depth_norm"
        self.transient_root = self.root_dir / "transient"

        self.pairs = []
        for obj_folder in sorted(self.depth_root.iterdir()):
            if not obj_folder.is_dir():
                continue
            obj_name = obj_folder.name
            transient_folder = self.transient_root / obj_name

            for depth_path in sorted(obj_folder.glob("*_view_*_norm.pfm")):
                stem = depth_path.stem.replace("_norm", "")
                trans_file = transient_folder / f"{stem}_transient.npy"
                if not trans_file.exists():
                    raise FileNotFoundError(f"No existe: {trans_file}")
                self.pairs.append((str(trans_file), str(depth_path)))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No se encontró ningún par en {root_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        trans_path, depth_path = self.pairs[idx]

        # 1) Cargo el .npy (shape original: H×W×T×3)
        I_np = np.load(trans_path).astype(np.float32)

        # 2) Si tiene canal RGB, descarto los 2 canales extra:
        #    paso de (32, 32, 256, 3) a (32, 32, 256)
        if I_np.ndim == 4:
            I_np = I_np[..., 0]

        # 3) Verifico forma tras descartar canal:
        if I_np.shape != (32, 32, 256):
            raise RuntimeError(f"[ERROR] Shape inesperada en {trans_path}: {I_np.shape}")

        # 4) Transpongo para que la dimensión temporal (256) quede en D:
        #    (H, W, T) → (T, H, W)  i.e. (256, 32, 32)
        I_np = np.transpose(I_np, (2, 0, 1))

        # 5) Agrego canal para Conv3d: (1, 256, 32, 32)
        I_np = np.expand_dims(I_np, axis=0)

        # 6) Paso a tensor
        I_tensor = torch.from_numpy(I_np)

        # 7) Cargo el depth map (.pfm) → (64, 64)
        O_np = load_pfm(depth_path)
        O_tensor = torch.from_numpy(O_np)

        return I_tensor, O_tensor
