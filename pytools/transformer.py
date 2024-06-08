# Standard library imports
import math

# Third-party imports
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, input_dims, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, input_dims) # Empty position matrix
        position = torch.arange(
            0, 
            max_seq_length, 
            dtype=torch.float
        ).unsqueeze(1) # Vector of each position in sequence
        div_term = torch.exp(
            torch.arange(
                0, 
                input_dims, 
                2
            ).float() * -(math.log(10000.0) / input_dims)
        ) 
        # Sine positional encoding for even dimensions
        pe[:, 0::2] = torch.sin(position * div_term) 
        # Cosine positional encoding for odd dimensions
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0)) # Allow GPU use

    def forward(self, x):
        pe = self.pe[:, :x.size(1)] # Get positional encoding for len(sequence)
        return x + pe # Add positional encoding to sequence


class SelfAttention(nn.Module):
    def __init__(self, input_dims, num_heads):
        super(SelfAttention, self).__init__()
        assert input_dims % num_heads == 0  # Can split into equal heads
        self.input_dims = input_dims
        self.num_heads = num_heads
        self.d_k = input_dims // num_heads  # Dimension of each head
        self.W_q = nn.Linear(input_dims, input_dims)  # Query
        self.W_k = nn.Linear(input_dims, input_dims)  # Key
        self.W_v = nn.Linear(input_dims, input_dims)  # Value
        self.W_o = nn.Linear(input_dims, input_dims)  # Output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scaling_factor = math.sqrt(self.d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scaling_factor
        if mask is not None: # Mask padded positions (set to -1e9)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probabilities = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probabilities, V)
        return output

    def split_heads(self, x):
        # Reshape input into (batch, seq_length, num_heads, dim_per_head)
        batch_size, seq_length, _ = x.size()
        return x.view(
            batch_size, 
            seq_length, 
            self.num_heads, 
            self.d_k
        ).transpose(1, 2) # Swap num_heads and seq_length

    def combine_heads(self, x):
        # Combine the last two dimensions into (input_dims)
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(
            batch_size, 
            seq_length, 
            self.input_dims
        )

    def forward(self, Q, K, V, mask=None):
        # Add layer of nodes then split into heads
        Q = self.split_heads(self.W_q(Q)) 
        K = self.split_heads(self.W_k(K)) 
        V = self.split_heads(self.W_v(V)) 
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attention_output)) # Combine heads
        return output


class FeedForward(nn.Module):
    def __init__(self, input_dims, ff_dims):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dims, ff_dims)
        self.fc2 = nn.Linear(ff_dims, input_dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Introduce dimensionality reduction with fc1
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, input_dims, num_heads, ff_dims, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(input_dims, num_heads)
        self.feed_forward = FeedForward(input_dims, ff_dims)
        self.norm = nn.LayerNorm(input_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask) # Self-attention
        # Add attention weight to input and normalize
        x = self.norm(x + self.dropout(attention_output)) 
        # Introduce non-linearity with feed-forward network
        ff_output = self.feed_forward(x)
        # Combine input with dimensionality-reduced output and normalize
        x = self.norm(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(
        self, vocab_size, input_dims, num_heads, num_layers, ff_dims, \
        max_seq_length, dropout
    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dims)
        self.positional_encoding = PositionalEncoding(
            input_dims, 
            max_seq_length
        )
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                input_dims,
                num_heads, 
                ff_dims, 
                dropout
            ) for _ in range(num_layers) # Add N encoder layers
        ])
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, input):
        mask = (input != 1).unsqueeze(1).unsqueeze(2)
        return mask 

    def forward(self, input):
        mask = self.generate_mask(input) # To zero out padding tokens
        embedding = self.embedding(input) # Vectorize input
        encoder_output = self.dropout(
            self.positional_encoding(embedding) # Add positional encoding
        )
        for encoder_layer in self.encoder_layers: # Feed through N layers
            encoder_output = encoder_layer(
                encoder_output, 
                mask
            ) # Self-attention mechanism
        pooled_output = torch.mean(encoder_output, dim=1) # Average pooling
        return pooled_output