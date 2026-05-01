# English-French Transformer Translator

This project is an educational English-French machine translation system built from scratch in PyTorch. It uses a custom Transformer architecture inspired by the original **"Attention Is All You Need"** paper.

The goal of this project was to understand how modern sequence-to-sequence models work internally, instead of only relying on pre-trained models.

## Features

- Custom multi-head attention
- Encoder and decoder blocks
- Positional encoding
- Padding and causal masks
- Training loop in PyTorch
- Basic inference script for translation

## My Contribution

I implemented the model architecture, prepared the data pipeline, trained the model, and debugged issues related to tensor shapes, masks, and loss calculation.

## Sample Outputs

```text
EN: Where did we come from?
FR: D'où venons-nous?

EN: What is light?
FR: Quest-ce que la lumière?

EN: Astronomes William Frederick King is born.
FR: Astronomes Naissance de William Frederick King.