import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
    context_window_size=None,
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
        context_window_size: Size of the context window for attention
        verbose: Whether to print progress
        
    Returns:
        model: Trained WalkTransformer model
    """
    vocab_size = len(vocab)
    max_seq_len = training_data.size(1)
    
    # Use context window size if provided, otherwise use max_seq_len
    effective_context_window = context_window_size if context_window_size is not None else max_seq_len
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"  Dataset size: {len(training_data)} walks")
        print(f"  Sequence length: {max_seq_len}")
        print(f"  Context window size: {effective_context_window}")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Number of epochs: {num_epochs}")
        print(f"  Device: {device}")
        print("\nModel Architecture:")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Number of heads: {num_heads}")
        print(f"  Dropout rate: {dropout}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max positional encoding: {max_seq_len}")
        print("="*60 + "\n")
    
    # Create model
    if verbose:
        print("Initializing model...")
    model = WalkTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create data loader
    if verbose:
        print("\nPreparing data loader...")
    dataset = TensorDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if verbose:
        print(f"  Number of batches: {len(dataloader)}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    if verbose:
        print("\nStarting training...")
        print("-" * 60)
    
    model.train()
    epoch_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        # Create progress bar for batches
        if verbose:
            batch_iterator = tqdm(dataloader, 
                                desc=f"Epoch {epoch+1}/{num_epochs}",
                                unit="batch",
                                leave=True)
        else:
            batch_iterator = dataloader
            
        for batch_idx, (batch,) in enumerate(batch_iterator):
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
            
            # Update progress bar
            if verbose:
                avg_loss = total_loss / (batch_idx + 1)
                batch_iterator.set_postfix({"avg_loss": f"{avg_loss:.4f}", 
                                           "curr_loss": f"{loss.item():.4f}"})
        
        avg_epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        if verbose:
            print(f"  Epoch {epoch+1} completed - Average loss: {avg_epoch_loss:.4f}")
            if epoch > 0:
                loss_change = epoch_losses[-1] - epoch_losses[-2]
                print(f"  Loss change: {loss_change:+.4f}")
            print()
    
    if verbose:
        print("="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"  Final loss: {epoch_losses[-1]:.4f}")
        if len(epoch_losses) > 1:
            total_improvement = epoch_losses[0] - epoch_losses[-1]
            print(f"  Total improvement: {total_improvement:.4f}")
            print(f"  Average loss per epoch: {sum(epoch_losses)/len(epoch_losses):.4f}")
        print("="*60 + "\n")
    
    return model
