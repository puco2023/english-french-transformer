import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq, d_model)
        return x + self.pe[:, :x.size(1)]

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model

        # Hand-built parameters for attention
        self.W_q = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_k = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_v = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_o = nn.Parameter(torch.Tensor(d_model, d_model))

        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_o)

        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout = nn.Dropout(dropout)

        assert d_model % nhead == 0

    def multi_head_attention(self, x, mask):
        # x: (batch, seq, d_model)
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        B, S, _ = x.shape
        H = self.nhead
        D = self.d_model // H

        # Reshape: (batch, seq, nhead, d_per_head) -> (batch, nhead, seq, d_per_head)
        Q = Q.view(B, S, H, D).transpose(1, 2)
        K = K.view(B, S, H, D).transpose(1, 2)
        V = V.view(B, S, H, D).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(D)
        mask = mask.unsqueeze(1).unsqueeze(1)  # (B,1,1,seq)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = attn @ V
        # (batch, nhead, seq, d_per_head) -> (batch, seq, d_model)
        context = context.transpose(1, 2).contiguous().view(B, S, self.d_model)
        output = context @ self.W_o
        return output

    def forward(self, x, mask):
        x = x + self.dropout(self.multi_head_attention(x, mask))
        x = self.ln1(x)
        x = x + self.dropout(self.feed_forward(x))
        x = self.ln2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model

        self.W_q = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_k = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_v = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_o = nn.Parameter(torch.Tensor(d_model, d_model))

        self.W_q_cross = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_k_cross = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_v_cross = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_o_cross = nn.Parameter(torch.Tensor(d_model, d_model))

        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_o)
        nn.init.xavier_uniform_(self.W_q_cross)
        nn.init.xavier_uniform_(self.W_k_cross)
        nn.init.xavier_uniform_(self.W_v_cross)
        nn.init.xavier_uniform_(self.W_o_cross)

        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout = nn.Dropout(dropout)

        assert d_model % nhead == 0

    def masked_self_attention(self, x, tgt_mask):
        B, S, _ = x.shape
        H = self.nhead
        D = self.d_model // H

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Q = Q.view(B, S, H, D).transpose(1, 2)
        K = K.view(B, S, H, D).transpose(1, 2)
        V = V.view(B, S, H, D).transpose(1, 2)

        # Look-ahead mask
        no_look_ahead_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        no_look_ahead_mask = ~no_look_ahead_mask  # True gde je dozvoljeno

        # tgt_mask: (B, S) -> (B, 1, 1, S)
        padding_mask = tgt_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
        combined_mask = no_look_ahead_mask.unsqueeze(0).unsqueeze(0) & padding_mask.expand(-1,1,S,-1)
        # (B,1,S,S)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(D)
        scores = scores.masked_fill(~combined_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        context = attn @ V
        context = context.transpose(1, 2).contiguous().view(B, S, self.d_model)
        output = context @ self.W_o
        return output

    def cross_attention(self, x, enc, src_mask):
        B, T, _ = x.shape
        S = enc.shape[1]
        H = self.nhead
        D = self.d_model // H

        Q = x @ self.W_q_cross
        K = enc @ self.W_k_cross
        V = enc @ self.W_v_cross

        Q = Q.view(B, T, H, D).transpose(1, 2)
        K = K.view(B, S, H, D).transpose(1, 2)
        V = V.view(B, S, H, D).transpose(1, 2)

        # src_mask: (B, S) -> (B,1,1,S)
        mask = src_mask.unsqueeze(1).unsqueeze(1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(D)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        context = attn @ V
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = context @ self.W_o_cross
        return output

    def forward(self, x, enc, tgt_mask, src_mask):
        x = x + self.dropout(self.masked_self_attention(x, tgt_mask))
        x = self.ln1(x)
        x = x + self.dropout(self.cross_attention(x, enc, src_mask))
        x = self.ln2(x)
        x = x + self.dropout(self.feed_forward(x))
        x = self.ln3(x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, vocab_size, max_len, nlayers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(nlayers)
        ])

    def forward(self, x, mask):
        x = self.dropout(self.pe(self.embedding(x)))
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, vocab_size, max_len, nlayers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(nlayers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, tgt_mask, src_mask):
        x = self.dropout(self.pe(self.embedding(x)))
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        x = self.linear(x)  # (batch, seq, vocab_size)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, vocab_size, max_len, nlayers, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(d_model, nhead, dim_feedforward, vocab_size, max_len, nlayers, dropout)
        self.decoder = Decoder(d_model, nhead, dim_feedforward, vocab_size, max_len, nlayers, dropout)

    def forward(self, src, tgt, tgt_mask, src_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return dec_output
