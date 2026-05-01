# English-French Transformer Translator

This project is an educational English-French machine translation system built from scratch in PyTorch. It uses a custom Transformer architecture inspired by the original **"Attention Is All You Need"** paper.

The goal of this project was to understand how modern sequence-to-sequence models work internally, instead of only relying on pre-trained models.

## Features

- Custom multi-head attention
- Encoder and decoder blocks
- Positional encoding
- Padding masks and causal masks
- Training loop in PyTorch
- Basic inference script for translation

## Project Structure

- `load_dataset.py` - dataset loading and tokenization
- `model.py` - custom Transformer implementation
- `training.py` - training loop
- `test.py` - inference/testing script
- `README.md` - project description
- `.gitignore` - ignored files

## My Contribution

I built the full training pipeline myself, including the Transformer model architecture, encoder and decoder blocks, multi-head attention, positional encoding, padding and causal masks, dataset loading, training loop, and inference script.

## Sample Outputs

```text
EN: Are we alone?
FR: Sommes-nouss?

EN: Where did we come from?
FR: D'où venons-nous?

EN: What is light ?
FR: Quest-ce que la lumière?

EN: Astronomes William Frederick King is born.
FR: Astronomes Naissance de William Frederick King.
```

## How to Run

Install the required libraries:

```bash
pip install torch pandas transformers
```

Place the English-French dataset CSV file in the project folder.

Train the model:

```bash
python training.py
```

Test the model after training:

```bash
python test.py
```

## Checkpoint and Dataset

The trained model checkpoint (`model.pt`) and the dataset are not included in this repository to keep the repository lightweight.

After training, `training.py` will generate a local `model.pt` checkpoint, which can then be used by `test.py` for inference.

## Note

This project is intended for learning and experimentation, not for production-level translation quality.
