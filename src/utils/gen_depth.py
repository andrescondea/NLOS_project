import os
import numpy as np
import mitsuba as mi
from pathlib import Path
import argparse


def save_pfm(image: np.ndarray, filename: str, scale: float = 1.0):
    """
    Guarda un array 2D (float32) en formato PFM.
    """
    with open(filename, 'wb') as f:
        f.write(b'Pf\n')
        height, width = image.shape
        f.write(f"{width} {height}\n".encode('utf-8'))
        endian = image.dtype.byteorder
        if endian == '<' or (endian == '=' and os.sys.byteorder == 'little'):
            scale = -scale
        f.write(f"{scale}\n".encode('utf-8'))
        image.tofile(f)


def generate_six_views_for_obj(obj_path: str, output_subdir: str):

    vistas = [
        {"rotate": None,                          "translate": [0.0, 0.0, 1.5]},
        {"rotate": {"axis": [0, 1, 0], "angle": 180}, "translate": [0.0, 0.0, -1.5]},
        {"rotate": {"axis": [0, 1, 0], "angle": 270}, "translate": [1.5, 0.0, 0.0]},
        {"rotate": {"axis": [1, 0, 0], "angle": 90},  "translate": [0.0, 1.5, 0.0]},
        {"rotate": {"axis": [1, 0, 0], "angle": 270}, "translate": [0.0, -1.5, 0.0]},
        {"rotate": None,                          "translate": [0.0, 0.0, 1.5]}
    ]

    base_name = Path(obj_path).stem  # p.ej. "model001"
    idx = 0
    for v in vistas:
        Tf = mi.ScalarTransform4f()
        if v["rotate"] is not None:
            ax = v["rotate"]["axis"]
            an = v["rotate"]["angle"]
            Tf = Tf.rotate(axis=ax, angle=an)
        tr = v["translate"]
        Tf = Tf.translate(mi.ScalarPoint3f(tr))

        # Construir escena para depth AOV
        scene_dict = {
            "type": "scene",
            "integrator": {"type": "aov", "aovs": "dd.z:depth"},
            "sensor": {
                "type": "perspective",
                "fov": 45,
                "to_world": mi.ScalarTransform4f().look_at(
                    [0, 0, 0.0],
                    [0, 0, 1.0],
                    [0, 1, 0]
                ),
                "film": {"type": "hdrfilm", "width": 64, "height": 64, "rfilter": {"type": "box"}},
                "sampler": {"type": "independent", "sample_count": 128}
            },
            "shape": {"type": "obj", "filename": obj_path, "to_world": Tf, "bsdf": {"type": "diffuse", "reflectance": 1.0}},
            "relay_wall": {"type": "rectangle", "to_world": mi.ScalarTransform4f().scale([1.0,1.0,1.0]).translate([0.0,0.0,0.0]), "bsdf": {"type": "diffuse", "reflectance": 1.0}}
        }

        scene = mi.load_dict(scene_dict)
        if scene is None:
            print(f"[ERROR] No se pudo cargar escena de '{obj_path}' en vista {idx}")
            idx += 1
            continue

        depth_tensor = mi.render(scene)            # ScalarFloat (64,64,1)
        depth_np = np.array(depth_tensor)[..., 0]   # (64,64)

        out_name = f"{base_name}_view_{idx:02d}.pfm"
        out_path = os.path.join(output_subdir, out_name)
        save_pfm(depth_np.astype(np.float32), out_path)
        print(f"Guardado: {out_path}")
        idx += 1


def main(input_dir: str, output_dir: str):
    mi.set_variant("scalar_rgb")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Buscar todos los .obj en input_dir
    obj_paths = sorted(Path(input_dir).glob("*.obj"))
    if not obj_paths:
        print(f"No se encontraron archivos .obj en {input_dir}")
        return

    for obj_path in obj_paths:
        print(f"Procesando '{obj_path.name}'...")
        # Para cada objeto, crear una subcarpeta con su nombre
        subfolder = Path(output_dir) / obj_path.stem
        subfolder.mkdir(exist_ok=True)
        generate_six_views_for_obj(str(obj_path), str(subfolder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera 6 vistas depth para cada .obj en una carpeta"
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Carpeta que contiene archivos .obj'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Carpeta donde se crearán subcarpetas con los depth .pfm'
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

# Ejemplo de uso:
# python gen_depth.py \
#   --input_dir /home/NLOS_project/objs \
#   --output_dir /home/NLOS_project/outputs/data_pfm_all
