import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import WalkTransformer


def train_model(
    training_data,
    vocab,
    epochs,
    batch_size,
    learning_rate,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
):
    model = WalkTransformer(
        len(vocab), d_model=512, nhead=8, num_layers=6, dim_feedforward=2048
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2idx["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(training_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            optimizer.zero_grad()
            output = model(batch[:, :-1])
            loss = criterion(
                output.contiguous().view(-1, len(vocab)),
                batch[:, 1:].contiguous().view(-1),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if verbose and (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model
