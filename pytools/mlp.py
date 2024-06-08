# Third-party imports
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, 
                 input_dims, 
                 num_layers, 
                 mlp_dims, 
                 num_classes, 
                 dropout):
        super(MLP, self).__init__()
        self.lin_layers = nn.ModuleList()
        if num_layers == 1 and type(mlp_dims) != list:
            mlp_dims = [mlp_dims]
        for i in range(num_layers):
            in_dims = input_dims if i == 0 else mlp_dims[i-1]
            out_dims = mlp_dims[i]
            self.lin_layers.append(
                nn.Linear(in_dims, out_dims)
            )
        
        self.out = nn.Linear(out_dims, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        for lin_layer in self.lin_layers:
            x = self.dropout(self.relu(lin_layer(x)))
        x = self.out(x)
        
        return x