import os
import numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
import mitransient as mitr
from pathlib import Path
import argparse

# Este script recorre todos los archivos .obj en una carpeta de entrada,
# y para cada .obj genera 6 vistas de imágenes transitorias (transients) guardándolos en la carpeta de salida.

def save_npy(volume: np.ndarray, filename: str):
    """
    Guarda un array 4D (H, W, T, C) en un archivo .npy.
    """
    np.save(filename, volume.astype(np.float32))


def generate_six_transients_for_obj(obj_path: str, output_subdir: str):
    """
    Dado un archivo .obj, genera 6 vistas transitorias con Mitsuba + mitransient y las guarda en output_subdir.
    """
    # Definir las seis transformaciones (rotaciones/traslaciones)
    vistas = [
        {"rotate": None,                          "translate": [0.0, 0.0, 1.5]},
        {"rotate": {"axis": [0, 1, 0], "angle": 180}, "translate": [0.0, 0.0, -1.5]},
        {"rotate": {"axis": [0, 1, 0], "angle": 270}, "translate": [1.5, 0.0, 0.0]},
        {"rotate": {"axis": [1, 0, 0], "angle": 90},  "translate": [0.0, 1.5, 0.0]},
        {"rotate": {"axis": [1, 0, 0], "angle": 270}, "translate": [0.0, -1.5, 0.0]},
        {"rotate": None,                          "translate": [0.0, 0.0, 1.5]}
    ]

    # Configuración común de Mitsuba + mitransient para volúmenes transitorios
    transient_film = {
        "type": "transient_hdr_film",
        "width": 32,
        "height": 32,
        "temporal_bins": 256,
        "bin_width_opl": 0.006,
        "start_opl": 3.0,
        "rfilter": {"type": "box"}
    }
    integrator_dict = {
        "type": "transient_nlos_path",
        "nlos_laser_sampling": True,
        "nlos_hidden_geometry_sampling": True,
        "nlos_hidden_geometry_sampling_do_rroulette": False,
        "temporal_filter": "box"
    }

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

        geometry = mi.load_dict({
            "type": "obj",
            "filename": obj_path,
            "to_world": Tf,
            "bsdf": {"type": "diffuse", "reflectance": 1.0}
        })
        emitter = mi.load_dict({
            "type": "projector",
            "irradiance": 100.0,
            "fov": 0.25,
            "to_world": mi.ScalarTransform4f().translate(mi.ScalarPoint3f([-0.5, 0.0, 0.25]))
        })
        nlos_sensor = mi.load_dict({
            "type": "nlos_capture_meter",
            "sampler": {"type": "independent", "sample_count": 25000},
            "account_first_and_last_bounces": False,
            "sensor_origin": [-0.5, 0.0, 0.25],
            "transient_film": transient_film
        })
        relay_wall = mi.load_dict({
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f().scale([1.0, 1.0, 1.0]).translate([0.0, 0.0, 0.0]),
            "bsdf": {"type": "diffuse", "reflectance": 1.0},
            "nlos_sensor": nlos_sensor
        })

        scene_dict = {
            "type": "scene",
            "integrator": integrator_dict,
            "emitter": emitter,
            "shape": geometry,
            "relay_wall": relay_wall
        }
        scene = mi.load_dict(scene_dict)
        if scene is None:
            print(f"[ERROR] No se pudo cargar escena de '{obj_path}' en vista {idx}")
            idx += 1
            continue

        # Ajustar foco al píxel central (32,32)
        pixel = mi.Point2f(32, 32)
        mitr.nlos.focus_emitter_at_relay_wall_pixel(pixel, relay_wall, emitter)

        # Renderizar transientes
        integrator = scene.integrator()
        data_steady, data_transient = integrator.render(scene)
        # data_transient tiene shape (32, 32, 256, 3)

        transient_np = np.array(data_transient)
        # Guardar .npy
        out_name = f"{base_name}_view_{idx:02d}_transient.npy"
        out_path = os.path.join(output_subdir, out_name)
        save_npy(transient_np, out_path)
        print(f"Guardado: {out_path}  (shape: {transient_np.shape})")
        idx += 1


def main(input_dir: str, output_dir: str):
    mi.set_variant('llvm_ad_rgb')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Buscar todos los .obj en input_dir
    obj_paths = sorted(Path(input_dir).glob("*.obj"))
    if not obj_paths:
        print(f"No se encontraron archivos .obj en {input_dir}")
        return

    for obj_path in obj_paths:
        print(f"Procesando '{obj_path.name}'...')")
        subfolder = Path(output_dir) / obj_path.stem
        subfolder.mkdir(exist_ok=True)
        generate_six_transients_for_obj(str(obj_path), str(subfolder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera 6 vistas transitorias para cada .obj en una carpeta"
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Carpeta que contiene archivos .obj'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Carpeta donde se crearán subcarpetas con los transients .npy'
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

# Ejemplo de uso:
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# python gen_transient.py \
#   --input_dir /home/NLOS_project/objs \
#   --output_dir /home/NLOS_project/outputs/transients_all
