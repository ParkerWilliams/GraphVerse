import torch


def seed_walk(
    model,
    start_sequence,
    max_length,
    vocab,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()
    current_sequence = start_sequence.copy()

    while len(current_sequence) < max_length:
        with torch.no_grad():
            input_tensor = (
                torch.tensor(
                    [vocab.token2idx[str(token)] for token in current_sequence]
                )
                .unsqueeze(0)
                .to(device)
            )
            output = model(input_tensor)
            next_token_idx = output[0, -1, :].argmax().item()
            next_token = vocab.idx2token[next_token_idx]

            if next_token == "<END>":
                break

            current_sequence.append(int(next_token))

    return current_sequence
