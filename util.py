import os
import pickle
import dbm
import pandas as pd

from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from sklearn.metrics import r2_score, roc_auc_score, average_precision_score

from scipy.stats import pearsonr

import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from model import CrossAffinity

class CustomDataset():
    def __init__(self, db_path_affinity, pdb_affinity):
        self.db_path_affinity = db_path_affinity
        
        self.dp = pdb_affinity
    
    def __len__(self):
        return len(self.dp)
    
    def __getitem__(self, idx):
        dp = self.dp[idx]
        with dbm.open(self.db_path_affinity, 'c') as db:
            return pickle.loads(db[self.dp[idx]])

def collate_fn(batch):
    data1_list, data2_list, y_list = [], [], []
    for (data1, data2, y) in batch:
        data1_list.append(data1)
        data2_list.append(data2)
        y_list.append(y)
    data1_batch = Data(batch=data1_list)
    data2_batch = Data(batch=data2_list)
    y_batch = torch.stack(y_list)
    return data1_batch, data2_batch, y_batch

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def split_list_into_folds(pdb_list, num_folds=5):
    length = len(pdb_list)
    fold_size = length // num_folds
    splits = []
    for i in range(num_folds):
        start = i * fold_size
        
        if i == num_folds - 1:
            end = length
        else:
            end = (i + 1) * fold_size
        splits.append(pdb_list[start:end])
    return splits

def train(model, train_dataloader, criterion_mse_affinity, criterion_huber_affinity, optimizer, device):
    model.train()
    epoch_num_samples_huber_affinity = 0
    epoch_batch_loss_huber_affinity = 0
    epoch_num_samples_mse_affinity = 0
    epoch_batch_loss_mse_affinity = 0
    y_pred_mse_affinity = []
    y_true_mse_affinity = []
    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        (data_1, data_2, y) = batch
        x_1, edge_index_1, batch_1 = data_1.x.to(device), data_1.edge_index.to(device), data_1.batch.to(device)
        x_2, edge_index_2, batch_2 = data_2.x.to(device), data_2.edge_index.to(device), data_2.batch.to(device)
        y = y.to(device).float()
        num_samples_huber_affinity = len(y)
        num_samples_mse_affinity = len(y)
        
        affinity_out = model(x_1, edge_index_1, batch_1, x_2, edge_index_2, batch_2).flatten()
        
        mse_loss_affinity = criterion_mse_affinity(affinity_out, y)
        huber_loss_affinity = criterion_huber_affinity(affinity_out, y)
        
        optimizer.zero_grad()
        huber_loss_affinity.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2, norm_type=1)
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_batch_loss_huber_affinity += huber_loss_affinity.item() * num_samples_huber_affinity
        epoch_num_samples_huber_affinity += num_samples_huber_affinity
        
        epoch_batch_loss_mse_affinity += mse_loss_affinity.item() * num_samples_mse_affinity
        epoch_num_samples_mse_affinity += num_samples_mse_affinity
        y_pred_mse_affinity += affinity_out.tolist()
        y_true_mse_affinity += y.tolist()
        
    epoch_batch_loss_huber_affinity = epoch_batch_loss_huber_affinity / epoch_num_samples_huber_affinity if epoch_num_samples_huber_affinity else 0
    epoch_batch_loss_mse_affinity = epoch_batch_loss_mse_affinity / epoch_num_samples_mse_affinity if epoch_num_samples_mse_affinity else 0
    
    r2_affinity = r2_score(y_true_mse_affinity, y_pred_mse_affinity) if y_true_mse_affinity and y_pred_mse_affinity else 0
    r_affinity = pearsonr(y_true_mse_affinity, y_pred_mse_affinity).statistic if y_true_mse_affinity and y_pred_mse_affinity else 0
    
    return epoch_batch_loss_huber_affinity, epoch_batch_loss_mse_affinity, r2_affinity, r_affinity

def test(model, test_dataloader, criterion_mse_affinity, criterion_huber_affinity, device):
    model.eval()
    epoch_num_samples_huber_affinity = 0
    epoch_batch_loss_huber_affinity = 0
    epoch_num_samples_mse_affinity = 0
    epoch_batch_loss_mse_affinity = 0
    y_pred_mse_affinity = []
    y_true_mse_affinity = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            (data_1, data_2, y) = batch
            x_1, edge_index_1, batch_1 = data_1.x.to(device), data_1.edge_index.to(device), data_1.batch.to(device)
            x_2, edge_index_2, batch_2 = data_2.x.to(device), data_2.edge_index.to(device), data_2.batch.to(device)
            y = y.to(device).float()
            num_samples_huber_affinity = len(y)
            num_samples_mse_affinity = len(y)
            
            affinity_out = model(x_1, edge_index_1, batch_1, x_2, edge_index_2, batch_2).flatten()
            
            mse_loss_affinity = criterion_mse_affinity(affinity_out, y)
            huber_loss_affinity = criterion_huber_affinity(affinity_out, y)
            
            epoch_batch_loss_huber_affinity += huber_loss_affinity.item() * num_samples_huber_affinity
            epoch_num_samples_huber_affinity += num_samples_huber_affinity
            
            epoch_batch_loss_mse_affinity += mse_loss_affinity.item() * num_samples_mse_affinity
            epoch_num_samples_mse_affinity += num_samples_mse_affinity
            y_pred_mse_affinity += affinity_out.tolist()
            y_true_mse_affinity += y.tolist()
    
    epoch_batch_loss_huber_affinity = epoch_batch_loss_huber_affinity / epoch_num_samples_huber_affinity if epoch_num_samples_huber_affinity else 0
    epoch_batch_loss_mse_affinity = epoch_batch_loss_mse_affinity / epoch_num_samples_mse_affinity if epoch_num_samples_mse_affinity else 0
    
    r2_affinity = r2_score(y_true_mse_affinity, y_pred_mse_affinity) if y_true_mse_affinity and y_pred_mse_affinity else 0
    r_affinity = pearsonr(y_true_mse_affinity, y_pred_mse_affinity).statistic if y_true_mse_affinity and y_pred_mse_affinity else 0
    
    return epoch_batch_loss_huber_affinity, epoch_batch_loss_mse_affinity, r2_affinity, r_affinity


def test_predict(model, test_dataloader, criterion_mse_affinity, criterion_huber_affinity, device):
    model.eval()
    epoch_num_samples_huber_affinity = 0
    epoch_batch_loss_huber_affinity = 0
    epoch_num_samples_mse_affinity = 0
    epoch_batch_loss_mse_affinity = 0
    y_pred_mse_affinity = []
    y_true_mse_affinity = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            (data_1, data_2, y) = batch
            x_1, edge_index_1, batch_1 = data_1.x.to(device), data_1.edge_index.to(device), data_1.batch.to(device)
            x_2, edge_index_2, batch_2 = data_2.x.to(device), data_2.edge_index.to(device), data_2.batch.to(device)
            y = y.to(device).float()
            num_samples_huber_affinity = len(y)
            num_samples_mse_affinity = len(y)
            
            affinity_out = model(x_1, edge_index_1, batch_1, x_2, edge_index_2, batch_2)
            
            mse_loss_affinity = criterion_mse_affinity(affinity_out, y)
            huber_loss_affinity = criterion_huber_affinity(affinity_out, y)
            
            epoch_batch_loss_huber_affinity += huber_loss_affinity.item() * num_samples_huber_affinity
            epoch_num_samples_huber_affinity += num_samples_huber_affinity
            
            epoch_batch_loss_mse_affinity += mse_loss_affinity.item() * num_samples_mse_affinity
            epoch_num_samples_mse_affinity += num_samples_mse_affinity
            y_pred_mse_affinity += affinity_out.tolist()
            y_true_mse_affinity += y.tolist()
            
    epoch_batch_loss_huber_affinity = epoch_batch_loss_huber_affinity / epoch_num_samples_huber_affinity if epoch_num_samples_huber_affinity else 0
    epoch_batch_loss_mse_affinity = epoch_batch_loss_mse_affinity / epoch_num_samples_mse_affinity if epoch_num_samples_mse_affinity else 0
    
    r2_affinity = r2_score(y_true_mse_affinity, y_pred_mse_affinity) if y_true_mse_affinity and y_pred_mse_affinity else 0
    r_affinity = pearsonr(y_true_mse_affinity, y_pred_mse_affinity).statistic if y_true_mse_affinity and y_pred_mse_affinity else 0
    
    return epoch_batch_loss_huber_affinity, epoch_batch_loss_mse_affinity, r2_affinity, r_affinity, y_pred_mse_affinity, y_true_mse_affinity

def collate_loss(model, epochs, train_dataloader, test_dataloader, criterion_mse_affinity, criterion_huber_affinity, optimizer, mlflow_experiment_id, weights_dir, cwd, device):
    model = model.to(device)
    early_stop_counter = 0
    for epoch in range(model.epoch, epochs + model.epoch):
        print(f'EPOCH: {epoch}')
        early_stop_counter += 1
        train_metrics = train(model, train_dataloader, criterion_mse_affinity, criterion_huber_affinity, optimizer, device)
        test_metrics = test(model, test_dataloader, criterion_mse_affinity, criterion_huber_affinity, device)
        epoch_batch_loss_huber_affinity_train, epoch_batch_loss_mse_affinity_train, r2_affinity_train, r_affinity_train = train_metrics
        epoch_batch_loss_huber_affinity_test, epoch_batch_loss_mse_affinity_test, r2_affinity_test, r_affinity_test = test_metrics
        rmse_affinity_train, rmse_affinity_test = np.sqrt(epoch_batch_loss_mse_affinity_train), np.sqrt(epoch_batch_loss_mse_affinity_test)
        
        mlflow.log_metric("train_huber_affinity_loss", epoch_batch_loss_huber_affinity_train, step=epoch)
        mlflow.log_metric("train_mse_affinity_loss", epoch_batch_loss_mse_affinity_train, step=epoch)
        mlflow.log_metric("train_rmse_affinity", rmse_affinity_train, step=epoch)
        mlflow.log_metric("train_r2_affinity", r2_affinity_train, step=epoch)
        mlflow.log_metric("train_r_affinity", r_affinity_train, step=epoch)
        mlflow.log_metric("valid_huber_affinity_loss", epoch_batch_loss_huber_affinity_test, step=epoch)
        mlflow.log_metric("valid_mse_affinity_loss", epoch_batch_loss_mse_affinity_test, step=epoch)
        mlflow.log_metric("valid_rmse_affinity", rmse_affinity_test, step=epoch)
        mlflow.log_metric("valid_r2_affinity", r2_affinity_test, step=epoch)
        mlflow.log_metric("valid_r_affinity", r_affinity_test, step=epoch)
        
        if model.best_r is None or r_affinity_test > model.best_r:
            model.best_r = r_affinity_test
            model.huber_train, model.mse_train, model.r2_train, model.r_train = epoch_batch_loss_huber_affinity_train, epoch_batch_loss_mse_affinity_train, r2_affinity_train, r_affinity_train
            model.huber_test, model.mse_test, model.r2_test = epoch_batch_loss_huber_affinity_test, epoch_batch_loss_mse_affinity_test, r2_affinity_test
            
            os.makedirs(os.path.join(cwd, weights_dir, mlflow_experiment_id), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cwd, weights_dir, mlflow_experiment_id, model.name + '.pth'))
            early_stop_counter = 0
        
        if early_stop_counter == 5:
            mlflow.log_param("stopped_epoch", epoch + 1)
            break
    
    scores = {'Best_Validation_R': [model.best_r], 'Huber_Train': [model.huber_train], 'MSE_Train': [model.mse_train], 'R2_Train': [model.r2_train], 'R_Train': [model.r_train], 'Huber_Test': [model.huber_test], 'MSE_Test': [model.mse_test], 'R2_Test': [model.r2_test]}
    
    scores = pd.DataFrame(scores)
    os.makedirs(os.path.join(cwd, weights_dir, mlflow_experiment_id), exist_ok=True)
    scores.to_csv(f'./mlruns/{mlflow_experiment_id}/{model.name}.csv', index=False)

def run(epochs, config, train_pdbs, test_pdbs, fold, device):
    inputdim = config['inputdim']
    num_gcn_layers = config['num_gcn_layers']
    fdim = config['fdim']
    dropout_rate = config['dropout_rate']
    num_recycle = config['num_recycle']
    num_heads = config['num_heads']
    lr = config['lr']
    mlflow_experiment_id = config['mlflow_experiment_id']
    weights_dir = config['weights_dir']
    cwd = config['cwd']
    batch_size = config['batch_size']
    train_dataset = CustomDataset(os.path.join(cwd, 'Train.dbm'), train_pdbs)
    test_dataset = CustomDataset(os.path.join(cwd, 'Train.dbm'), test_pdbs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    name = f'{inputdim}_{fdim}_{num_gcn_layers}_{num_recycle}_{num_heads}_{batch_size}_{dropout_rate}_{lr}_{fold}'
    device = torch.device(device)
    
    runs = mlflow.search_runs(
        experiment_ids=[mlflow_experiment_id],
    )
    
    if runs.shape[0] != 0:
        if f"{name}_fold_{fold}" not in runs['tags.mlflow.runName'].tolist():
            finished = False
        else:
            finished = True
    else:
        finished = False
    
    if not finished:
        with mlflow.start_run(run_name=f"{name}_fold_{fold}", experiment_id=mlflow_experiment_id):
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("fdim", fdim)
            mlflow.log_param("num_heads", num_heads)
            mlflow.log_param("num_recycle", num_recycle)
            mlflow.log_param("num_gcn_layers", num_gcn_layers)
            mlflow.log_param("inputdim", inputdim)
            
            model = CrossAffinity(name, inputdim=1280, fdim=fdim, num_gcn_layers=num_gcn_layers,
                                  dropout_rate=dropout_rate, num_heads=num_heads, num_recycle=num_recycle, 
                                  ).to(device)
            model.apply(weights_init)
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            criterion_huber_affinity = nn.HuberLoss()
            criterion_mse_affinity = nn.MSELoss()
            criterion_bce_map = nn.BCELoss()
            criterion_bce_ppi = nn.BCELoss()
            criterion_bce_ppi_res = nn.BCELoss()
            collate_loss(model, epochs, train_dataloader, test_dataloader, criterion_mse_affinity, criterion_huber_affinity, optimizer, mlflow_experiment_id, weights_dir, cwd, device)