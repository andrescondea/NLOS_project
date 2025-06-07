import os
import yaml
import torch
import json

from torch.utils.data import DataLoader
from src.data.dataset import NLoSDataset
from src.models.depthmap_model import DepthMap
from src.utils.metrics import rmse, accuracy_fg_bg

def evaluate_model(model, loader, device):
    model.eval()
    total_rmse = 0.0
    total_acc  = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for I_batch, O_batch in loader:
            I_batch = I_batch.to(device)
            O_batch = O_batch.to(device)
            
            pred = model(I_batch)  # (B, H, W)
            batch_size = I_batch.size(0)
            
            # Sumamos la métrica multiplicada por número de muestras
            total_rmse += rmse(pred, O_batch).item() * batch_size
            total_acc  += accuracy_fg_bg(pred, O_batch).item() * batch_size
            num_samples += batch_size
    
    avg_rmse = total_rmse / num_samples
    avg_acc  = total_acc  / num_samples
    return avg_rmse, avg_acc

def main(config_path: str, checkpoint_path: str):
    # Leer configuración
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear DataLoader para test
    raw_root    = cfg['data']['raw_root']       
    test_subfld = cfg['data']['test_subfolder']   
    test_dir    = os.path.join(raw_root, test_subfld)
    batch_size  = cfg['evaluation']['batch_size']  
    
    test_dataset = NLoSDataset(
        raw_dir             = test_dir,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # Instanciar modelo y cargar pesos
    model = DepthMap().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    
    # Evaluar en test
    rmse_val, acc_val = evaluate_model(model, test_loader, device)
    print('>>> Test results:')
    print(f'    RMSE (foreground): {rmse_val:.4f}')
    print(f'    Accuracy (fg/bg):  {acc_val:.4f}')
    
    # Guardar reporte
    output = {
        'rmse_fg': rmse_val,
        'accuracy_fg_bg': acc_val
    }
    save_dir = os.path.dirname(checkpoint_path)
    save_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f'Results saved to {save_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate DepthMap NLoS')
    parser.add_argument('--config',     type=str, required=True,
                        help='Ruta al YAML de configuración para evaluación (configs/evaluate.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Ruta al modelo entrenado (ej. experiments/exp01_baseline/best_model.pth)')
    args = parser.parse_args()
    main(args.config, args.checkpoint)


""" python src/evaluate.py \
    --config configs/evaluate.yaml \
    --checkpoint experiments/exp01_baseline/best_model.pth """