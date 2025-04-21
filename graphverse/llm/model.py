import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class WalkTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout, max_seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = nn.Embedding(max_seq_len, hidden_size)  # Learnable positional encoding
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4*hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Create position indices
        positions = torch.arange(src.size(1), device=src.device).unsqueeze(0).expand(src.size(0), -1)
        
        # Get embeddings and positional encodings
        embedded = self.embedding(src)
        pos_encoding = self.pos_encoder(positions)
        
        # Combine embeddings and positional encodings
        x = embedded + pos_encoding
        x = self.dropout(x)
        
        # Pass through transformer
        output = self.transformer(x)
        
        # Project to vocabulary size
        return self.fc_out(output)
