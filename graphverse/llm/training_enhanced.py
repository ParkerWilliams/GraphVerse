import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import math

from graphverse.llm.model_enhanced import EnhancedWalkTransformer


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss to prevent overconfidence."""
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size * seq_len, vocab_size]
            targets: [batch_size * seq_len]
        """
        # Create smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            
            # Zero out padding positions
            mask = targets == self.padding_idx
            true_dist[mask] = 0
        
        # Calculate KL divergence loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        
        # Mask out padding
        loss = loss.masked_fill(mask, 0)
        
        # Return mean loss over non-padding tokens
        return loss.sum() / (~mask).sum()


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine annealing."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * (self.min_lr_ratio + 
                                (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def train_model_enhanced(
    training_data,
    vocab,
    hidden_size=384,
    num_layers=4,
    num_heads=6,
    dropout=0.1,
    batch_size=64,
    num_epochs=20,
    learning_rate=0.001,
    context_window_size=16,
    verbose=True,
    label_smoothing=0.1,
    warmup_ratio=0.1,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
    weight_decay=0.01
):
    """Enhanced training function with warmup, label smoothing, and better optimization."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Get vocab size and max sequence length
    vocab_size = len(vocab)
    max_seq_len = training_data.shape[1]
    
    if verbose:
        print("\n" + "="*60)
        print("ENHANCED TRAINING CONFIGURATION")
        print("="*60)
        print(f"  Dataset size: {len(training_data):,} walks")
        print(f"  Sequence length: {max_seq_len}")
        print(f"  Context window size: {context_window_size}")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps} steps")
        print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"  Number of epochs: {num_epochs}")
        print(f"  Device: {device}")
        print("\nModel Architecture:")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Number of heads: {num_heads}")
        print(f"  Feedforward dim: 2048")
        print(f"  Dropout rate: {dropout}")
        print(f"  Label smoothing: {label_smoothing}")
        print("\nOptimization:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Warmup ratio: {warmup_ratio}")
        print(f"  Max gradient norm: {max_grad_norm}")
        print("="*60 + "\n")
    
    # Create enhanced model
    if verbose:
        print("Initializing enhanced transformer model...")
    
    model = EnhancedWalkTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=max_seq_len,
        dim_feedforward=2048,
        use_temperature=True,
        label_smoothing=label_smoothing
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
    num_batches = len(dataloader)
    if verbose:
        print(f"  Number of batches: {num_batches}")
    
    # Loss function with label smoothing
    if label_smoothing > 0:
        criterion = LabelSmoothingLoss(vocab_size, vocab.token2idx["<PAD>"], smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx["<PAD>"])
    
    # Optimizer with weight decay (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup
    total_steps = num_batches * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    if verbose:
        print(f"\nScheduler configuration:")
        print(f"  Total training steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
    
    # Training loop
    if verbose:
        print("\nStarting enhanced training...")
        print("-" * 60)
    
    model.train()
    epoch_losses = []
    global_step = 0
    current_lr = learning_rate  # Initialize current learning rate
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_tokens = 0
        
        # Create progress bar
        if verbose:
            batch_iterator = tqdm(dataloader, 
                                desc=f"Epoch {epoch+1}/{num_epochs}",
                                unit="batch",
                                leave=True)
        else:
            batch_iterator = dataloader
        
        optimizer.zero_grad()
        
        for batch_idx, (batch,) in enumerate(batch_iterator):
            batch = batch.to(device)
            
            # Split into input and target
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]
            
            # Create padding mask
            padding_mask = (input_seq == vocab.token2idx["<PAD>"])
            
            # Forward pass
            logits = model(input_seq, src_key_padding_mask=padding_mask)
            
            # Reshape for loss calculation
            B, T, V = logits.shape
            loss = criterion(logits.reshape(-1, V), target_seq.reshape(-1))
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Update learning rate
                current_lr = scheduler.step()
                global_step += 1
            
            # Track loss
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Count non-padding tokens
            num_tokens += (~padding_mask).sum().item()
            
            # Update progress bar
            if verbose:
                avg_loss = total_loss / (batch_idx + 1)
                perplexity = math.exp(min(avg_loss, 100))  # Cap perplexity to prevent overflow
                batch_iterator.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "ppl": f"{perplexity:.2f}",
                    "lr": f"{current_lr:.6f}"
                })
        
        # Calculate epoch metrics
        avg_epoch_loss = total_loss / len(dataloader)
        epoch_perplexity = math.exp(min(avg_epoch_loss, 100))
        epoch_losses.append(avg_epoch_loss)
        
        if verbose:
            print(f"  Epoch {epoch+1} completed - Loss: {avg_epoch_loss:.4f}, Perplexity: {epoch_perplexity:.2f}")
            if epoch > 0:
                loss_change = epoch_losses[-1] - epoch_losses[-2]
                print(f"  Loss change: {loss_change:+.4f}")
            print()
    
    if verbose:
        print("="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"  Final loss: {epoch_losses[-1]:.4f}")
        print(f"  Final perplexity: {math.exp(min(epoch_losses[-1], 100)):.2f}")
        if len(epoch_losses) > 1:
            total_improvement = epoch_losses[0] - epoch_losses[-1]
            print(f"  Total improvement: {total_improvement:.4f}")
            print(f"  Average loss per epoch: {sum(epoch_losses)/len(epoch_losses):.4f}")
        print("="*60 + "\n")
    
    return model