# Third-party imports
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, 
                 device,
                 batch_size, 
                 vocab_size,
                 max_seq_length, 
                 num_layers, 
                 cnn_layer_dims, 
                 kernel_size, 
                 pooling_size, 
                 stride, 
                 dropout,
                 embedding_dims=None):
        super(CNN, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            vocab_size, embedding_dims
        ).to(self.device) if embedding_dims is not None else None
        self.embedding_dims = embedding_dims if (
            embedding_dims is not None
        ) else vocab_size
        self.max_seq_length = max_seq_length

        self.conv_layers = nn.ModuleList()
        if num_layers == 1:
            if type(cnn_layer_dims) != list:
                cnn_layer_dims = [cnn_layer_dims]
            if type(kernel_size) != list:
                kernel_size = [kernel_size]
        for i in range(num_layers):
            in_channels = self.embedding_dims if i == 0 else cnn_layer_dims[i-1]
            out_channels = cnn_layer_dims[i]
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size[i],
                    stride=stride,
                    padding=(kernel_size[i] - 1) // 2
                ).to(device)
            )

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(pooling_size)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
    
    def get_output_dims(self):       
        if self.embedding is not None:
            tmp = torch.zeros(
                (self.batch_size, self.max_seq_length)
            )
            tmp = tmp.to(self.device, torch.long)
            tmp = self.embedding(tmp)
        else:
            tmp = torch.zeros(
                (self.batch_size, self.max_seq_length, self.vocab_size)
            )
            tmp = tmp.to(self.device, torch.float)
        tmp = tmp.permute(0, 2, 1)
        for conv_layer in self.conv_layers:
            tmp = conv_layer(tmp)
            tmp = self.relu(tmp)
        tmp = self.maxpool(tmp)

        return tmp.view(self.batch_size, -1).shape[1]
    
    def forward(self, x):
        if self.embedding is not None:
            x = x.to(self.device, torch.long)
            x = self.embedding(x)
        else:
            x = x.to(self.device, torch.float)
        x = x.permute(0, 2, 1)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dropout(x)

        return x