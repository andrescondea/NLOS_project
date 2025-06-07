import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import yaml
import random
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from utils.dataset import NLoSDataset
from models.depthmap import DepthMap
from src.utils.losses import get_loss
from src.utils.metrics import rmse, accuracy_fg_bg

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for I_batch, O_batch in loader:
        print("DEBUG: I_batch.shape =", I_batch.shape, "  O_batch.shape =", O_batch.shape)
        I_batch = I_batch.to(device)      # shape (B, 1, T, H, W) o (B, 1, D, H, W)
        O_batch = O_batch.to(device)      # shape (B, H_o, W_o)
        
        optimizer.zero_grad()
        pred = model(I_batch)             # shape (B, H_o, W_o)
        loss = criterion(pred, O_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * I_batch.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    val_acc  = 0.0
    with torch.no_grad():
        for I_batch, O_batch in loader:
            I_batch = I_batch.to(device)
            O_batch = O_batch.to(device)
            pred = model(I_batch)
            loss = criterion(pred, O_batch)
            
            val_loss += loss.item() * I_batch.size(0)
            val_rmse += rmse(pred, O_batch).item() * I_batch.size(0)
            val_acc  += accuracy_fg_bg(pred, O_batch).item() * I_batch.size(0)
    
    num_samples = len(loader.dataset)
    avg_loss = val_loss / num_samples
    avg_rmse = val_rmse / num_samples
    avg_acc  = val_acc  / num_samples
    return avg_loss, avg_rmse, avg_acc

def main(config_path: str):
    # Leer configuración YAML
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set seed y device
    seed   = cfg['training'].get('seed', 42)
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parámetros básicos
    raw_root   = cfg['data']['raw_root']       # ej. "data/raw"
    train_dir  = os.path.join(raw_root, cfg['data']['train_subfolder'])  # ej. "data/raw/train"
    val_dir    = os.path.join(raw_root, cfg['data']['val_subfolder'])    # ej. "data/raw/val"
    batch_size = cfg['training']['batch_size'] # ej. 8
    num_epochs = cfg['training']['num_epochs'] # ej. 30
    lr         = cfg['training']['lr']         # ej. 1e-4
    loss_name  = cfg['training']['loss']       # ej. "mse"
    exp_dir    = cfg['experiment']['exp_dir']  # ej. "experiments/exp01_baseline"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Dataset y DataLoader
    train_dataset = NLoSDataset("/home/agrosavia/Desktop/MitsLight/data/raw/train")
    val_dataset   = NLoSDataset("/home/agrosavia/Desktop/MitsLight/data/raw/val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    
    # Modelo, loss, optimizador
    model     = DepthMap().to(device)
    criterion = get_loss(loss_name)
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Archivo para registrar métricas por época
    log_path = os.path.join(exp_dir, 'train_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_rmse,val_acc\n')
    
    best_val_loss = float('inf')
    
    # Bucle de entrenamiento
    for epoch in range(1, num_epochs+1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        # Guardar métricas en CSV
        with open(log_path, 'a') as f:
            f.write(f'{epoch},{train_loss:.6f},'
                    f'{val_loss:.6f},{val_rmse:.6f},{val_acc:.6f}\n')
        
        print(f'Epoch {epoch}/{num_epochs} | '
              f'TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | '
              f'ValRMSE: {val_rmse:.4f} | ValAcc: {val_acc:.4f}')
        
        # Guardar checkpoint de estado completo cada época
        ckpt_path = os.path.join(exp_dir, f'checkpoint_epoch{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, ckpt_path)
        
        # Si mejoró la loss de validación, guardamos “best_model.pth”
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(exp_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train DepthMap NLoS')
    parser.add_argument('--config', type=str, required=True,
                        help='Ruta al archivo YAML de configuración (p. ej. configs/train.yaml)')
    args = parser.parse_args()
    main(args.config)

""" 
cd mi_proyecto
python -m venv venv                     # o tu método preferido de entorno virtual
source venv/bin/activate
pip install -r requirements.txt        # incluyendo torch, numpy, pyyaml, etc.

python src/train.py --config configs/train.yaml """