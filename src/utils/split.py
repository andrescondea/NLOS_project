import os
import shutil
import random
from pathlib import Path
import argparse

def split_folders_by_object(depth_root: Path,
                            transient_root: Path,
                            output_root: Path,
                            val_fraction: float = 0.2,
                            seed: int = 42):

    random.seed(seed)

    depth_subdirs = sorted([d.name for d in depth_root.iterdir() if d.is_dir()])
    transient_subdirs = sorted([d.name for d in transient_root.iterdir() if d.is_dir()])

    if set(depth_subdirs) != set(transient_subdirs):
        diff1 = set(depth_subdirs) - set(transient_subdirs)
        diff2 = set(transient_subdirs) - set(depth_subdirs)
        raise RuntimeError(f"Las subcarpetas no coinciden:\n"
                           f"  en depth_norm pero no en transient: {diff1}\n"
                           f"  en transient pero no en depth_norm: {diff2}")

    all_objects = depth_subdirs  # lista de nombres
    random.shuffle(all_objects)

    n_total = len(all_objects)
    n_val = int(round(val_fraction * n_total))
    val_objs = set(all_objects[:n_val])
    train_objs = set(all_objects[n_val:])

    print(f"Total de objetos encontrados: {n_total}")
    print(f" → Asignando {len(val_objs)} a VALIDACIÓN, {len(train_objs)} a TRAIN.")

    for split_name, obj_set in zip(["train", "val"], [train_objs, val_objs]):
        for kind in ["depth_norm", "transient"]:
            dst_base = output_root / split_name / kind
            dst_base.mkdir(parents=True, exist_ok=True)

            src_base = depth_root if kind == "depth_norm" else transient_root

            for obj in obj_set:
                src_folder = src_base / obj
                dst_folder = dst_base / obj
                if dst_folder.exists():
                    shutil.rmtree(dst_folder)
                shutil.copytree(src_folder, dst_folder)

    print("Partición completada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Divide depth_norm/ y transient/ en train/ y val/ manteniendo subcarpetas por objeto"
    )
    parser.add_argument(
        "--depth_root", type=str, required=True,
        help="Carpeta depth_norm/"
    )
    parser.add_argument(
        "--transient_root", type=str, required=True,
        help="Carpeta transient/"
    )
    parser.add_argument(
        "--output_root", type=str, required=True,
        help="Carpeta donde crear data/train/ y data/val/"
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.2,
        help="Fracción de objetos que irán a validación (entre 0 y 1)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla para barajar objetos de forma reproducible."
    )
    args = parser.parse_args()

    split_folders_by_object(
        depth_root    = Path(args.depth_root),
        transient_root= Path(args.transient_root),
        output_root   = Path(args.output_root),
        val_fraction  = args.val_fraction,
        seed          = args.seed
    )
