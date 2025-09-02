import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with optional learnable component."""
    def __init__(self, d_model, max_len=5000, learnable=True):
        super().__init__()
        
        # Create sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Optional learnable component
        if learnable:
            self.learnable_pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        else:
            self.learnable_pe = None
    
    def forward(self, x):
        """x shape: [batch_size, seq_len, d_model]"""
        seq_len = x.size(1)
        
        # Handle sequences longer than max_len by repeating last position
        if seq_len > self.pe.size(1):
            # Clone and extend with last position repeated
            pos_encoding = torch.cat([
                self.pe[:, :self.pe.size(1), :],
                self.pe[:, -1:, :].expand(-1, seq_len - self.pe.size(1), -1)
            ], dim=1)
        else:
            pos_encoding = self.pe[:, :seq_len, :].clone()
        
        if self.learnable_pe is not None:
            if seq_len > self.learnable_pe.size(1):
                # Similarly handle learnable component
                learnable_encoding = torch.cat([
                    self.learnable_pe[:, :self.learnable_pe.size(1), :],
                    self.learnable_pe[:, -1:, :].expand(-1, seq_len - self.learnable_pe.size(1), -1)
                ], dim=1)
                pos_encoding = pos_encoding + learnable_encoding
            else:
                pos_encoding = pos_encoding + self.learnable_pe[:, :seq_len, :]
        
        return x + pos_encoding


class EnhancedTransformerLayer(nn.Module):
    """Enhanced transformer layer with pre-norm and gated residuals."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network with ReLU
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),  # Using ReLU instead of GELU
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Learnable gates for residual connections
        self.gate1 = nn.Parameter(torch.ones(1))
        self.gate2 = nn.Parameter(torch.ones(1))
    
    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Pre-norm and self-attention with gated residual
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, 
                                     attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask)
        x = x + self.gate1 * self.dropout1(attn_out)
        
        # Pre-norm and feedforward with gated residual
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + self.gate2 * self.dropout2(ff_out)
        
        return x


class EnhancedWalkTransformer(nn.Module):
    """Enhanced transformer for graph walk prediction with improved architecture."""
    
    def __init__(self, vocab_size, hidden_size=384, num_layers=4, num_heads=6, 
                 dropout=0.1, max_seq_len=100, dim_feedforward=2048,
                 use_temperature=True, label_smoothing=0.0):
        super().__init__()
        
        # Validate heads divide hidden size
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.label_smoothing = label_smoothing
        
        # Token and positional embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size, max_seq_len, learnable=True)
        
        # Embedding dropout
        self.emb_dropout = nn.Dropout(dropout)
        
        # Stack of enhanced transformer layers
        self.layers = nn.ModuleList([
            EnhancedTransformerLayer(hidden_size, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size)
        
        # Output projection with optional temperature
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        # Learnable temperature for softmax
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.temperature = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization for better gradient flow."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Special initialization for embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Initialize output projection with smaller weights
        nn.init.normal_(self.fc_out.weight, mean=0, std=0.02)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: [batch_size, seq_len] token indices
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        x = self.embedding(src) * math.sqrt(self.hidden_size)  # Scale embeddings
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.emb_dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.fc_out(x)
        
        # Apply temperature scaling if enabled
        if self.temperature is not None:
            logits = logits / self.temperature.clamp(min=0.1)  # Prevent division by zero
        
        return logits
    
    def get_attention_weights(self, src):
        """Get attention weights for visualization."""
        attention_weights = []
        x = self.embedding(src) * math.sqrt(self.hidden_size)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            # This would need modification of EnhancedTransformerLayer to return attention weights
            # For now, returning empty list
            pass
        
        return attention_weights


# Backward compatibility - keep old model name available
WalkTransformer = EnhancedWalkTransformer