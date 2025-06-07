import re
import numpy as np
import matplotlib.pyplot as plt

def load_pfm(file_path):
    """
    Lee un archivo PFM y devuelve un array numpy float32 de shape (H, W).
    """
    with open(file_path, 'rb') as f:
        header = f.readline().decode('latin-1').rstrip()
        if header != 'Pf':
            raise Exception('No es un archivo PFM (se esperaba "Pf" al inicio).')
        
        dim_line = f.readline().decode('latin-1').rstrip()
        m = re.match(r'^(\d+)\s+(\d+)\s*$', dim_line)
        if not m:
            raise Exception('Formato de dimensiones incorrecto en PFM.')
        width, height = map(int, m.groups())
        
        scale_line = f.readline().decode('latin-1').rstrip()
        scale = float(scale_line)
        endian = '<' if scale < 0 else '>'
        if scale < 0:
            scale = -scale
        
        data = np.fromfile(f, endian + 'f')
        expected = width * height
        if data.size != expected:
            raise Exception(f'Número de elementos inesperado: {data.size}, esperaba {expected}.')
        
        data = np.reshape(data, (height, width))
        # En PFM los datos se almacenan de abajo hacia arriba
        data = np.flipud(data)
        return data.astype(np.float32)

def inspect_pfm(file_path):
    """
    Carga el PFM, imprime estadísticas y muestra la imagen en pantalla.
    """
    depth = load_pfm(file_path)
    print(f"Archivo: {file_path}")
    print(f"Shape: {depth.shape}")
    print(f"Min: {depth.min():.6f}   Max: {depth.max():.6f}   Mean: {depth.mean():.6f}")
    
    plt.figure(figsize=(4,4))
    plt.imshow(depth, cmap='viridis')
    plt.title(f"Vista depth")
    plt.colorbar(label="Profundidad")
    plt.axis('off')
    plt.show()

# Ejemplo de uso:
inspect_pfm('/home/agrosavia/Desktop/MitsLight/data/data_test/arbol.pfm')
