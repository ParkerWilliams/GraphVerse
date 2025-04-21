import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import WalkTransformer


def train_model(
    training_data,
    vocab,
    hidden_size=256,
    num_layers=4,
    num_heads=8,
    dropout=0.1,
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False
):
    """
    Train a transformer model on the walk data.
    
    Args:
        training_data: Tensor of shape (N, max_seq_len) containing input sequences
        vocab: WalkVocabulary object mapping node indices to tokens
        hidden_size: Size of transformer hidden layers
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        model: Trained WalkTransformer model
    """
    vocab_size = len(vocab)
    max_seq_len = training_data.size(1)
    
    # Create model
    model = WalkTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)
    
    # Create data loader
    dataset = TensorDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(device)
            
            # Create input and target sequences
            input_seq = batch[:, :-1]  # All tokens except last
            target_seq = batch[:, 1:]  # All tokens except first
            
            # Forward pass
            logits = model(input_seq)
            
            # Reshape for loss calculation
            B, T, V = logits.shape
            loss = criterion(logits.reshape(-1, V), target_seq.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if verbose and (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Avg Loss: {avg_loss:.4f}")
        
        if verbose:
            avg_epoch_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} completed, Avg Loss: {avg_epoch_loss:.4f}")
    
    return model
