import torch
from torch import nn

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, sequential
from torch_geometric.utils import to_dense_batch, scatter, softmax

from copy import copy

class GCNLayers(torch.nn.Module):
    def __init__(
            self,
            inputdim=1280,
            fdim=256,
            num_gcn_layers=1,
                 ):
        
        super(GCNLayers, self).__init__()
        assert num_gcn_layers > 0, "Layers need to be more than 0"
        
        self.inputdim = inputdim
        self.fdim = fdim
        self.num_gcn_layers = num_gcn_layers

        self.linear = nn.Sequential(
                          nn.Linear(inputdim, fdim),
                          nn.LayerNorm(fdim),
                          nn.LeakyReLU()
                      )
        
        self.gcn_layers = nn.ModuleList([GCNConv(fdim, fdim) for _ in range(self.num_gcn_layers)])
        self.activation_layers = nn.ModuleList([nn.Sequential(
                                                  nn.LayerNorm(fdim),
                                                  nn.LeakyReLU()
                                              ) for _ in range(self.num_gcn_layers)])
                
    def forward(self, x_1, edge_index_1, x_2, edge_index_2):
        x_1 = self.linear(x_1)
        x_2 = self.linear(x_2)
        
        for idx in range(self.num_gcn_layers):
            x_1 = self.gcn_layers[idx](x_1, edge_index_1)
            x_2 = self.gcn_layers[idx](x_2, edge_index_2)
            
            x_1 = self.activation_layers[idx](x_1)
            x_2 = self.activation_layers[idx](x_2)
        
        return x_1, x_2
        

class AttentionCross(torch.nn.Module):
    def __init__(
            self,
            fdim=256, 
            num_heads=1, 
            num_recycle=2,
            dropout_rate=0.1
            ):
        
        super(AttentionCross, self).__init__()
        self.fdim = fdim
        self.num_heads = num_heads
        self.num_recycle = num_recycle

        self.k = nn.Sequential(
                        nn.Linear(fdim, fdim),
                        nn.LayerNorm(fdim),
                        nn.LeakyReLU()
                )
        self.q = nn.Sequential(
                        nn.Linear(fdim, fdim),
                        nn.LayerNorm(fdim),
                        nn.LeakyReLU()
                )
        self.v = nn.Sequential(
                        nn.Linear(fdim, fdim),
                        nn.LayerNorm(fdim),
                        nn.LeakyReLU()
                )
        
        self.cross_attn = nn.MultiheadAttention(fdim, num_heads, dropout_rate, batch_first=True)

    def forward(self, x_1, batch_1, x_2, batch_2):
        
        for _ in range(self.num_recycle):
            x_1_copy = copy(x_1)
            x_2_copy = copy(x_2)
            
            x_1_k = self.k(x_1)
            x_1_q = self.q(x_1)
            x_1_v = self.v(x_1)
            x_2_k = self.k(x_2)
            x_2_q = self.q(x_2)
            x_2_v = self.v(x_2)
            
            x_1_padded_k, mask_1 = to_dense_batch(x_1_k, batch_1)
            x_1_padded_q, mask_1 = to_dense_batch(x_1_q, batch_1)
            x_1_padded_v, mask_1 = to_dense_batch(x_1_v, batch_1)
            x_2_padded_k, mask_2 = to_dense_batch(x_2_k, batch_2)
            x_2_padded_q, mask_2 = to_dense_batch(x_2_q, batch_2)
            x_2_padded_v, mask_2 = to_dense_batch(x_2_v, batch_2)
            
            x_1, attn_x1 = self.cross_attn(x_1_padded_q, x_2_padded_k, x_2_padded_v, key_padding_mask=~mask_2)
            x_2, attn_x2 = self.cross_attn(x_2_padded_q, x_1_padded_k, x_1_padded_v, key_padding_mask=~mask_1)
            
            x_1 = x_1[mask_1] + x_1_copy
            x_2 = x_2[mask_2] + x_2_copy
        
        return x_1, x_2

class SimpleAttentionPool(torch.nn.Module):
    def __init__(self, fdim=256):
        super(SimpleAttentionPool, self).__init__()
        self.attn_score = nn.Sequential(
                        nn.Linear(fdim, 1),
                        nn.LeakyReLU()
                )

    def forward(self, x_1, batch_1, x_2, batch_2):
        x_1_attn_score = self.attn_score(x_1)
        x_2_attn_score = self.attn_score(x_2)
        
        x_1_attn_score = softmax(x_1, batch_1)
        x_2_attn_score = softmax(x_2, batch_2)
        return x_1_attn_score, x_2_attn_score


class CrossAffinity(torch.nn.Module):
    def __init__(self, name, inputdim=1280, fdim=256, num_gcn_layers=1, dropout_rate=0.1, num_heads=1, num_recycle=2):
        super(CrossAffinity, self).__init__()
        
        self.name = name
        self.inputdim = inputdim
        self.fdim = fdim
        self.dropout_rate = dropout_rate
        self.num_recycle = num_recycle

        self.epoch = 0
        self.best_r = None
        
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn_layer = GCNLayers(inputdim, fdim, num_gcn_layers)
        self.attn_cross = AttentionCross(fdim, num_heads, num_recycle, dropout_rate)

        self.attn_pool = SimpleAttentionPool(fdim)
        
        self.linear = nn.Sequential(
                        nn.Linear(fdim, fdim),
                        nn.LayerNorm(fdim),
                        nn.LeakyReLU()
                )

        self.reduce_dim = nn.Sequential(
                        nn.Linear(4 * fdim, fdim),
                        nn.LeakyReLU(),
                        nn.LayerNorm(fdim)
                )
        
        self.affinity_layer = nn.Sequential(
                        nn.Linear(fdim, fdim),
                        nn.LeakyReLU(),
                        nn.LayerNorm(fdim),
                        nn.Linear(fdim, 1),
                        nn.LeakyReLU(negative_slope=1.2)
                )
    
    def forward(self, x_1, edge_index_1, batch_1, x_2, edge_index_2, batch_2):
        x_1, x_2 = self.gcn_layer(x_1, edge_index_1, x_2, edge_index_2)
        x_1 = self.dropout(x_1)
        x_2 = self.dropout(x_2)
        
        x_1, x_2 = self.attn_cross(x_1, batch_1, x_2, batch_2)

        x_1 = self.dropout(x_1)
        x_2 = self.dropout(x_2)

        x_1 = self.linear(x_1)
        x_2 = self.linear(x_2)
        
        x_1_attn_score, x_2_attn_score = self.attn_pool(x_1, batch_1, x_2, batch_2)
        
        x_1_attn_pool = self.global_attn_pool(x_1, x_1_attn_score, batch_1)
        mean_pool_1 = global_mean_pool(x_1, batch_1)
        max_pool_1 = global_max_pool(x_1, batch_1)
        min_pool_1 = self.global_min_pool(x_1, batch_1)
        pooled_1 = torch.cat([mean_pool_1, max_pool_1, min_pool_1, x_1_attn_pool], dim=1)

        x_2_attn_pool = self.global_attn_pool(x_2, x_2_attn_score, batch_2)
        mean_pool_2 = global_mean_pool(x_2, batch_2)
        max_pool_2 = global_max_pool(x_2, batch_2)
        min_pool_2 = self.global_min_pool(x_2, batch_2)
        pooled_2 = torch.cat([mean_pool_2, max_pool_2, min_pool_2, x_2_attn_pool], dim=1)

        plus_pooled = (pooled_1 + pooled_2) / 2

        plus_pooled = self.dropout(self.reduce_dim(plus_pooled))

        affinity_out = self.affinity_layer(plus_pooled)
        
        return affinity_out
    
    def global_min_pool(self, x, batch):
        dim = -1 if isinstance(x, torch.Tensor) and x.dim() == 1 else -2
        if batch is None:
            return x.min(dim=dim, keepdim=x.dim() <= 2)[0]
        return scatter(x, batch, dim=dim, reduce='min')
    
    def global_attn_pool(self, x, x_attn_score, batch):
        attn_pool = scatter(x * x_attn_score, batch, dim=0, reduce="sum")

        return attn_pool