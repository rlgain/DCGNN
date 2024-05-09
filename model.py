import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

inf = math.inf

class BiLSTM(nn.Module):
    def __init__(self, n_feat, out_dim, rnn_layers, dropout):
        super(BiLSTM, self).__init__()
        self.trans = nn.LSTM(
            input_size=n_feat,
            hidden_size=out_dim,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, X):
        x_hidden, _ = self.trans(X)
        output = x_hidden[:, -1, :]
        return output

class CausalAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(CausalAggregator, self).__init__()
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        self.weight = nn.Parameter(torch.Tensor(input_dim+3, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature, Beta):
        n_stocks = input_feature.size(0)
        output = torch.zeros(n_stocks, self.output_dim)
        for y_id in range(n_stocks):
            source = torch.cat([input_feature, adjacency[:, y_id, :]], dim=1)
            u = torch.mm(source, self.weight)
            output[y_id] = Beta[:, y_id].unsqueeze(0).mm(u).squeeze()

        if self.use_bias:
            output += self.bias
        return output
    
class DCGNN(nn.Module):
    def __init__(self, n_feat, n_out=2, emb_dim=64, dropout=0.3):
        super(DCGNN, self).__init__()
        dn = 384
        du = 128

        self.theta = np.load('./theta.npy') # Theta is the standard deviation of all stocks' historical returns
        self.lambdaa = 0.38

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.rnn = BiLSTM(n_feat=n_feat, out_dim=int(dn/2), rnn_layers=2, dropout=dropout)
        self.CausalAgg1 = CausalAggregator(input_dim=dn, output_dim=du)
        self.CausalAgg2 = CausalAggregator(input_dim=du, output_dim=du)
        self.out = nn.Linear(dn+du+du, n_out, bias=True)

        self.all_PP = self._load_pattern_prototypes_()

    def _load_pattern_prototypes_(self):
        self.all_PP = np.load('./PC_x2y_nasdaq100_t2t+1_14-16.npy')
        return self.all_PP
    
    def _dynamic_network_inference_(self, input):
        n_stocks = input.size(0)
        E = 3
        adj = torch.zeros(n_stocks, n_stocks, 3)
        Beta = torch.zeros(n_stocks, n_stocks)
        signature = [-1, 0, 1] # ↘，→，↗
        patterns = list(product(signature, repeat=(E-1)))

        for x_id, X in enumerate(input):
            x_return_series = X[-(E-1):,0]
            Px = tuple(np.where(x_return_series > self.theta[x_id], 1, np.where(x_return_series < -self.theta[x_id], -1, 0)))
            for pattern_id, pattern in enumerate(patterns):
                if pattern == Px:
                    break

            for y_id, Y in enumerate(input):
                y_return_series = Y[-(E-2):,0]
                Py = np.where(y_return_series > self.theta[y_id], 1, np.where(y_return_series < -self.theta[y_id], -1, 0))
                if Py == -1: # ↘
                    e = self.all_PP[x_id, y_id, pattern_id, 0:3]
                if Py == 0: # →
                    e = self.all_PP[x_id, y_id, pattern_id, 3:6]
                if Py == 1: # ↗
                    e = self.all_PP[x_id, y_id, pattern_id, 6:9]
                adj[x_id, y_id] = torch.Tensor(e)
                beta = torch.norm(torch.Tensor(e), p=2)
                Beta[x_id, y_id] = torch.Tensor(np.where(beta > self.lambdaa, beta, 0))

        Beta = F.softmax(Beta, dim=0)
        return adj, Beta

    def forward(self, input):
        adj, Beta = self._dynamic_network_inference_(input)
        n = self.rnn(input)
        n1 = self.CausalAgg1(adj, n, Beta)
        n2 = self.CausalAgg2(adj, n1, Beta)
        m = self.out(torch.cat([n,n1,n2], dim=1))
        y_hat = torch.tanh(m)
        return y_hat
    
## ----------------- Loss Function --------------------
    
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()

    def forward(self, x, y):
        pred_loss = F.cross_entropy(input=x, target=y)
        total_loss = pred_loss
        return total_loss