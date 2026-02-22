import os
import pandas as pd
import esm
import dbm
import pickle
import multiprocessing
import torch
import torch.nn as nn
import math
from torch_geometric.data import Data, DataLoader
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse

worker_lock = None

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
        sequence_representations = []
        for i, (seq, tokens_len) in enumerate(zip(batch_strs, batch_lens)):
            token_map[seq] = [token_representations[i, 1 : tokens_len - 1].to('cpu')]
        
        for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
            token_map[seq].append(attention_contacts[: tokens_len, : tokens_len].to('cpu'))
        
    with dbm.open('ESM2_Train.dbm', 'c') as db:
        for key, val in tqdm(token_map.items()):
            db[key] = pickle.dumps(val)



def prepare_single(part_1_sequences, part_2_sequences, pKd, dp_num):
    protein_1 = part_1_sequences.split(';')
    protein_2 = part_2_sequences.split(';')
    
    idx_1 = 0
    x_1 = []
    edge_index_1 = []
    
    for chain_idx, seq in enumerate(protein_1):
        with worker_lock:
            with dbm.open('ESM2_Train.dbm', 'c') as db:
                embed, contact_map = pickle.loads(db[seq])
        
        wanted_edge_idx = torch.nonzero(contact_map > 0.5)
        edge_indices = (wanted_edge_idx.t() + idx_1).contiguous()
        edge_attrs = torch.tensor([[contact_map[x, y].item()] for x, y in wanted_edge_idx.tolist()])
        
        
        
        x_1.append(embed)
        edge_index_1.append(edge_indices)
        
        idx_1 += len(seq)
    
    idx_2 = 0
    x_2 = []
    edge_index_2 = []
    
    for chain_idx, seq in enumerate(protein_2):
        with worker_lock:
            with dbm.open('ESM2_Train.dbm', 'c') as db:
                embed, contact_map = pickle.loads(db[seq])
        
        wanted_edge_idx = torch.nonzero(contact_map > 0.5)
        edge_indices = (wanted_edge_idx.t() + idx_2).contiguous()
        edge_attrs = torch.tensor([[contact_map[x, y].item()] for x, y in wanted_edge_idx.tolist()])
        
        
        
        x_2.append(embed)
        edge_index_2.append(edge_indices)
        
        idx_2 += len(seq)
    
    x_1 = torch.cat(x_1)
    x_2 = torch.cat(x_2)
    
    edge_index_1 = torch.cat(edge_index_1, 1)
    edge_index_2 = torch.cat(edge_index_2, 1)
    
    data_1 = Data(x=x_1, edge_index=edge_index_1, part_1_sequences=part_1_sequences)
    data_2 = Data(x=x_2, edge_index=edge_index_2, part_2_sequences=part_2_sequences)
    y = torch.tensor(pKd)
    
    with worker_lock:
        with dbm.open('Train.dbm', 'c') as db:
            db[dp_num] = pickle.dumps((data_1, data_2, y))

def prepare(part_1, part_2, pKds, esm2_device, num_workers):
    sequences = list(set([y for x in part_1 for y in x.split(';')] + [y for x in part_2 for y in x.split(';')]))
    print('Running ESM2...')
    esm_embedding(sequences, esm2_device)
    print('Preparing datapoints...')
    dp_nums = [str(i) for i in range(len(part_1))]
    with multiprocessing.Manager() as manager:
        lock = manager.Lock()
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(lock,)) as executor:
            _ = list(executor.map(prepare_single, part_1, part_2, pKds, dp_nums))
    

def affinity(pdb, train_df):
    unit = train_df.loc[pdb, 'unit']
    value = train_df.loc[pdb, 'value']
    if unit == 'mM':
        value = value * (10**-3)
    elif unit == 'uM':
        value = value * (10**-6)
    elif unit == 'nM':
        value = value * (10**-9)
    elif unit == 'pM':
        value = value * (10**-12)
    elif unit == 'fM':
        value = value * (10**-15)
    
    pKd = -math.log(value, 10)
    return pKd

def parse_csv(esm2_device, num_workers=1):
    train_df = pd.read_excel('training.xlsx')
    train_df.index = train_df['PDB ID']
    rcsb_df = pd.read_csv('RCSB_PDB_Table.csv')
    rcsb_df.index = rcsb_df.apply(lambda x: f'{x["PDB ID"]}_{x["Chain"]}', axis=1)
    
    part_1 = []
    part_2 = []
    pKds = []
    for _, row in train_df.iterrows():
        part_1_seq = ';'.join([rcsb_df.loc[f'{row["PDB ID"]}_{i}', 'Sequence'] for i in row['pairwise composition'].replace(' ', '').replace(',','').strip(';').split(';')[0]])
        part_2_seq = ';'.join([rcsb_df.loc[f'{row["PDB ID"]}_{i}', 'Sequence'] for i in row['pairwise composition'].replace(' ', '').replace(',','').strip(';').split(';')[1]])
        part_1.append(part_1_seq)
        part_2.append(part_2_seq)
        
        pKd = affinity(row['PDB ID'], train_df)
        pKds.append(pKd)
    
    prepare(part_1, part_2, pKds, esm2_device, num_workers)
    os.remove('ESM2_Train.dbm.dat')
    os.remove('ESM2_Train.dbm.bak')
    os.remove('ESM2_Train.dbm.dir')

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Preparation script")
    parser.add_argument("--esm2_device", type=str, help="Device to use ESM2 on. e.g. 'cpu', 'cuda:0'")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for parallelisation.")
    
    args = parser.parse_args()
    
    parse_csv(args.esm2_device, args.num_workers)