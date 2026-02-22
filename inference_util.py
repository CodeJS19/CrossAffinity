import os
import pandas as pd
import esm
import dbm
import pickle
import multiprocessing
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from model import CrossAffinity

worker_lock = None

class CustomDataset():
    def __init__(self, db_path_affinity, pdb_affinity):
        self.db_path_affinity = db_path_affinity
        
        # Create a single list of all data points
        self.dp = pdb_affinity
    
    def __len__(self):
        return len(self.dp)
    
    def __getitem__(self, idx):
        dp = self.dp[idx]
        with dbm.open(self.db_path_affinity, 'c') as db:
            return pickle.loads(db[self.dp[idx]])


def init_worker(l):
    global worker_lock
    worker_lock = l

def collate_fn(batch):
    data1_list, data2_list = [], []
    for (data1, data2) in batch:
        data1_list.append(data1)
        data2_list.append(data2)
    data1_batch = Data(batch=data1_list)
    data2_batch = Data(batch=data2_list)
    return data1_batch, data2_batch

def esm_embedding(sequences, esm2_device):
    token_map = {}
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    
    model = model.to(esm2_device)
    model.eval()
    
    for idx in tqdm(range(len(sequences))):
        data = [(sequences[idx], sequences[idx])]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(esm2_device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        
        del results['logits']
        del results['attentions']
        
        token_representations = results["representations"][33]
        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, (seq, tokens_len) in enumerate(zip(batch_strs, batch_lens)):
            token_map[seq] = [token_representations[i, 1 : tokens_len - 1].to('cpu')]
        
        # unsupervised self-attention map contact predictions
        for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
            token_map[seq].append(attention_contacts[: tokens_len, : tokens_len].to('cpu'))
        
    with dbm.open('ESM2.dbm', 'c') as db:
        for key, val in tqdm(token_map.items()):
            db[key] = pickle.dumps(val)



def prepare_single(part_1_sequences, part_2_sequences, dp_num):
    protein_1 = part_1_sequences.split(';')
    protein_2 = part_2_sequences.split(';')
    
    idx_1 = 0
    x_1 = []
    edge_index_1 = []
    edge_attr_1 = []
    subgraph_idx_1 = []
    
    for chain_idx, seq in enumerate(protein_1):
        with worker_lock:
            with dbm.open('ESM2.dbm', 'c') as db:
                embed, contact_map = pickle.loads(db[seq])
        
        wanted_edge_idx = torch.nonzero(contact_map > 0.5)
        edge_indices = (wanted_edge_idx.t() + idx_1).contiguous()
        edge_attrs = torch.tensor([[contact_map[x, y].item()] for x, y in wanted_edge_idx.tolist()])
        
        
        
        x_1.append(embed)
        edge_index_1.append(edge_indices)
        edge_attr_1.append(edge_attrs)
        subgraph_idx_1 = subgraph_idx_1 + [chain_idx] * len(seq)
        
        idx_1 += len(seq)
    
    idx_2 = 0
    x_2 = []
    edge_index_2 = []
    edge_attr_2 = []
    subgraph_idx_2 = []
    
    for chain_idx, seq in enumerate(protein_2):
        with dbm.open('ESM2.dbm', 'c') as db:
            embed, contact_map = pickle.loads(db[seq])
        
        wanted_edge_idx = torch.nonzero(contact_map > 0.5)
        edge_indices = (wanted_edge_idx.t() + idx_2).contiguous()
        edge_attrs = torch.tensor([[contact_map[x, y].item()] for x, y in wanted_edge_idx.tolist()])
        
        
        
        x_2.append(embed)
        edge_index_2.append(edge_indices)
        edge_attr_2.append(edge_attrs)
        subgraph_idx_2 = subgraph_idx_2 + [chain_idx] * len(seq)
        
        idx_2 += len(seq)
    
    x_1 = torch.cat(x_1)
    x_2 = torch.cat(x_2)
    
    edge_index_1 = torch.cat(edge_index_1, 1)
    edge_index_2 = torch.cat(edge_index_2, 1)
    
    edge_attr_1 = torch.cat(edge_attr_1)
    edge_attr_2 = torch.cat(edge_attr_2)
    
    subgraph_idx_1 = torch.tensor(subgraph_idx_1)
    subgraph_idx_2 = torch.tensor(subgraph_idx_2)
    
    
    data_1 = Data(x=x_1, edge_index=edge_index_1, edge_attr=edge_attr_1, subgraph_idx=subgraph_idx_1, part_1_sequences=part_1_sequences)
    data_2 = Data(x=x_2, edge_index=edge_index_2, edge_attr=edge_attr_2, subgraph_idx=subgraph_idx_2, part_2_sequences=part_2_sequences)
    
    with worker_lock:
        with dbm.open('Inference.dbm', 'c') as db:
            db[dp_num] = pickle.dumps((data_1, data_2))

def prepare(part_1, part_2, esm2_device, num_workers):
    sequences = list(set([y for x in part_1 for y in x.split(';')] + [y for x in part_2 for y in x.split(';')]))
    print('Running ESM2...')
    esm_embedding(sequences, esm2_device)
    dp_nums = [str(i) for i in list(range(len(part_1)))]
    print('Preparing datapoints...')
    with multiprocessing.Manager() as manager:
        shared_lock = manager.Lock()
        with ProcessPoolExecutor(num_workers, initializer=init_worker, initargs=(shared_lock,)) as executor:
            _ = list(executor.map(prepare_single, part_1, part_2, dp_nums))
    
    return dp_nums


def run_model(model, cross_affinity_device, dataloader):
    model.eval()
    results = {'Part 1':[], 'Part 2':[], model.name:[]}
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            (data_1, data_2) = batch
            x_1, edge_index_1, batch_1 = data_1.x.to(cross_affinity_device), data_1.edge_index.to(cross_affinity_device), data_1.batch.to(cross_affinity_device)
            x_2, edge_index_2, batch_2 = data_2.x.to(cross_affinity_device), data_2.edge_index.to(cross_affinity_device), data_2.batch.to(cross_affinity_device)
            affinity_out = model(x_1, edge_index_1, batch_1, x_2, edge_index_2, batch_2)
            affinity_out = [y for x in affinity_out.tolist() for y in x]
            
            results['Part 1'] += data_1.part_1_sequences
            results['Part 2'] += data_2.part_2_sequences
            results[model.name] += affinity_out
    
    results = pd.DataFrame(results)
    return results

def inference(model_path, model_name, part_1, part_2, cross_affinity_device, dataloader, output):
    model_results = []
    inputdim, fdim, num_gcn_layers, num_recycle, num_heads, batch_size, dropout_rate, lr = model_name.split('_')
    inputdim, fdim, num_gcn_layers, dropout_rate, num_recycle, num_heads = int(inputdim), int(fdim), int(num_gcn_layers), float(dropout_rate), int(num_recycle), int(num_heads)
    if dropout_rate == 0:
        dropout_rate = 0
    
    print('Predicting pKd...')
    for fold in range(1, 6):
        name = f'{inputdim}_{fdim}_{num_gcn_layers}_{num_recycle}_{num_heads}_{batch_size}_{dropout_rate}_{lr}_{fold}'
        
        model = CrossAffinity(name, inputdim=1280, fdim=fdim, num_gcn_layers=num_gcn_layers,
                              dropout_rate=dropout_rate, num_heads=num_heads, num_recycle=num_recycle, 
                              ).to(cross_affinity_device)
        state_dict = torch.load(f'{model_path}/{name}.pth', map_location=cross_affinity_device)
        model.load_state_dict(state_dict)
        results = run_model(model, cross_affinity_device, dataloader)
        model_results.append(results)
    
    all_dfs = []
    for sub_df in model_results:
        sub_df.index = sub_df.apply(lambda x: (x['Part 1'], x['Part 2']), axis=1)
        all_dfs.append(sub_df[[sub_df.columns[-1]]])
    
    all_dfs.insert(0, sub_df[['Part 1', 'Part 2']])
    all_dfs = pd.concat(all_dfs, axis=1)
    all_dfs['pKd'] = all_dfs[all_dfs.columns[2:]].mean(axis=1)
    
    all_dfs.to_csv(output, index=False)

def parse_csv(filepath, batch_size, cross_affinity_device, esm2_device, model_path, model_name, num_workers=1, output='Predicted_pKd.csv'):
    df = pd.read_csv(filepath, header=None)
    part_1 = df[0].tolist()
    part_2 = df[1].tolist()
    
    dp_nums = prepare(part_1, part_2, esm2_device, num_workers)
    
    dataset = CustomDataset('Inference.dbm', dp_nums)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    inference(model_path, model_name, part_1, part_2, cross_affinity_device, dataloader, output)
    
    os.remove('ESM2.dbm.dat')
    os.remove('ESM2.dbm.bak')
    os.remove('ESM2.dbm.dir')
    os.remove('Inference.dbm.dat')
    os.remove('Inference.dbm.bak')
    os.remove('Inference.dbm.dir')