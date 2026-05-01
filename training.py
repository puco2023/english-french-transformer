from load_dataset import data_loader, tokenizer, MAX_LENGTH
from model import Transformer

import torch
import torch.optim as optim
import torch.nn as nn
import os

num_epoch = 300
d_model = 128
nhead = 4
dim_feedforward = 256
nlayers = 1

vocab_size = len(tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Vocab size:", vocab_size)
print("Dataset batches:", len(data_loader))


model = Transformer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    vocab_size=vocab_size,
    max_len=MAX_LENGTH,
    nlayers=nlayers
).to(device)


criterion = nn.CrossEntropyLoss(ignore_index=-100)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)


min_loss = float("inf")

for epoch in range(num_epoch):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        src = batch["src"].to(device)
        src_mask = batch["src_mask"].to(device).bool()

        tgt_input = batch["tgt_input"].to(device)
        tgt_mask = batch["tgt_mask"].to(device).bool()
        tgt_output = batch["tgt_output"].to(device)

        optimizer.zero_grad()

        output = model(src, tgt_input, tgt_mask, src_mask)

        loss = criterion(
            output.reshape(-1, vocab_size),
            tgt_output.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epoch}] "
                f"Batch [{batch_idx}/{len(data_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(data_loader)

    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    if avg_loss < min_loss:
        min_loss = avg_loss

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "nlayers": nlayers,
            "max_len": MAX_LENGTH,
            "loss": min_loss
        }

        torch.save(checkpoint, "model.pt")
        print(f"Saved new best model with loss: {min_loss:.4f}")