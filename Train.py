import os
from itertools import product, repeat
import pandas as pd
import dbm.dumb as dbm
import random
import sys
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from concurrent.futures import ThreadPoolExecutor
import argparse

from util import *

exp_base_name = 'CrossAffinity'
epochs = 150
num_folds = 5

with dbm.open('Train.dbm', 'c') as db:
    all_pdbs = [i.decode() for i in db.keys()]

random.shuffle(all_pdbs)

splits = split_list_into_folds(all_pdbs, num_folds=num_folds)

all_train_pdbs = []
all_test_pdbs = []
fold_num = list(range(1, num_folds + 1))

for i in range(num_folds):
    all_test_pdbs.append(splits[i])
    all_train_pdbs.append(list(set(all_pdbs) - set(splits[i])))

cwd = os.getcwd()

client = MlflowClient()

created = 0
for i in range(100):
    try:
        exp_name = exp_base_name + "_{}".format(i)
        experiment_id = client.create_experiment(exp_name)
        created = 1
        break
    except (TypeError, Exception):
        continue

if not created:
    print("ERROR: Try new experiment name.")
    sys.exit(1)

mlflow.set_tracking_uri(f"file://{cwd}/mlruns/")
os.makedirs(f'mlruns/{experiment_id}', exist_ok=True)
shutil.copyfile(__file__, f'mlruns/{experiment_id}/{os.path.basename(__file__)}')

weights_root = './model_weights/'
weights_dir = weights_root + exp_name + '/'
os.makedirs(weights_dir, exist_ok=True)

grid_space = {
    'lr': [0.0001, 0.001],
    'fdim': [64, 128],
    'batch_size': [16, 32],
    'dropout_rate': [0, 0.2],
}

keys = grid_space.keys()
values = grid_space.values()
combinations = [dict(zip(keys, v)) for v in product(*values)]

all_combinations = []
for config in combinations:
    config['mlflow_experiment_id'] = experiment_id
    config['weights_dir'] = weights_dir
    config['cwd'] = cwd
    config['num_gcn_layers'] = 1
    config['inputdim'] = 1280
    config['num_heads'] = 2
    config['num_recycle'] = 2
    all_combinations += [config] * num_folds


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--device", type=str, help="Device to train on. e.g. 'cpu', 'cuda:0'")
    
    args = parser.parse_args()
    for a, b, c, d in zip(all_combinations, all_train_pdbs * len(combinations), all_test_pdbs * len(combinations), fold_num * len(combinations)):
        run(epochs, a, b, c, d, args.device)