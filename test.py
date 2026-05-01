from model import Transformer
from transformers import AutoTokenizer

import torch


MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
MODEL_PATH = "model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


checkpoint = torch.load(MODEL_PATH, map_location=device)

model = Transformer(
    d_model=checkpoint["d_model"],
    nhead=checkpoint["nhead"],
    dim_feedforward=checkpoint["dim_feedforward"],
    vocab_size=checkpoint["vocab_size"],
    max_len=checkpoint["max_len"],
    nlayers=checkpoint["nlayers"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded model.")
print("Best training loss:", checkpoint["loss"])


def translate(sentence, max_new_tokens=50):
    src_enc = tokenizer(
        [sentence],
        truncation=True,
        padding=True,
        max_length=checkpoint["max_len"],
        return_tensors="pt"
    )

    src = src_enc.input_ids.to(device)
    src_mask = src_enc.attention_mask.to(device).bool()

    generated = torch.tensor(
        [[tokenizer.pad_token_id]],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        for _ in range(max_new_tokens):
            tgt_mask = torch.ones_like(generated, device=device).bool()

            output = model(src, generated, tgt_mask, src_mask)

            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            # Jednostavna zaštita od beskonačnog ponavljanja
            if generated.shape[1] > 8:
                last_5 = generated[0, -5:].tolist()
                if len(set(last_5)) == 1:
                    break

    translation = tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )

    return translation


test_sentences = [
    "Are we alone?",
    "Who are we?",
    "Where did we come from?",
    "What is light ?",
    "Astronomes William Frederick King is born."
]

for sentence in test_sentences:
    print()
    print("EN:", sentence)
    print("FR:", translate(sentence))

