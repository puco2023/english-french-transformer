import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset


CSV_PATH = "eng-fra-part-real51.csv"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
MAX_LENGTH = 128
BATCH_SIZE = 32


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


class TranslationDataset(Dataset):
    def __init__(self, csv_path=CSV_PATH, max_rows=None):
        
        try:
            self.df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            self.df = pd.read_csv(csv_path, encoding="utf-8-sig")

        self.df = self.df.dropna(subset=["en", "fr"])
        self.df["en"] = self.df["en"].astype(str)
        self.df["fr"] = self.df["fr"].astype(str)

        if max_rows is not None:
            self.df = self.df.head(max_rows).reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "en": row["en"],
            "fr": row["fr"]
        }

    def __len__(self):
        return len(self.df)


def collate_fn(batch):
    eng = [item["en"] for item in batch]
    fra = [item["fr"] for item in batch]

    eng_enc = tokenizer(
        eng,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    fra_enc = tokenizer(
        fra,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    B = fra_enc.input_ids.shape[0]

    bos = torch.full(
        (B, 1),
        tokenizer.pad_token_id,
        dtype=torch.long
    )

    bos_mask = torch.ones(B, 1, dtype=torch.long)

    tgt_with_bos = torch.cat([bos, fra_enc.input_ids], dim=1)
    mask_with_bos = torch.cat([bos_mask, fra_enc.attention_mask], dim=1)

    tgt_input = tgt_with_bos[:, :-1]
    tgt_output = tgt_with_bos[:, 1:].clone()
    tgt_mask = mask_with_bos[:, :-1]
    tgt_output[fra_enc.attention_mask == 0] = -100

    return {
        "src": eng_enc.input_ids,
        "src_mask": eng_enc.attention_mask,
        "tgt_input": tgt_input,
        "tgt_mask": tgt_mask,
        "tgt_output": tgt_output
    }


df = TranslationDataset()

data_loader = DataLoader(
    df,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)