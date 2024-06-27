import math

import torch

from torch import nn, Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            emb_size: int,
            dropout: float,
            maxlen: int = 128, # TODO tune/allow to pass arg
            ):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size) # pylint: disable=E1101
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) # pylint: disable=E1101
        pos_embedding = torch.zeros((maxlen, emb_size)) # pylint: disable=E1101
        pos_embedding[:, 0::2] = torch.sin(pos * den) # pylint: disable=E1101
        pos_embedding[:, 1::2] = torch.cos(pos * den) # pylint: disable=E1101
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TransformerDecoder(nn.Module):
    '''
    Decode sequence, such as the representation of a string by CACNINE.
    Uses a vanilla transformer decoder architecture.

    '''
    def __init__(
            self,
            num_decoder_layers: int,
            emb_size: int,
            nhead: int,
            vocab_size: int,
            dim_feedforward: int = 512,
            dropout: float = 0.1,
            ):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            )

        self.token_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.head = nn.Linear(emb_size, vocab_size)

    def forward_features( # pylint: disable=C0116
            self,
            memory: Tensor, # i.e., input encoded by CANINE
            target: Tensor,
            target_mask: Tensor,
            target_padding_mask: Tensor,
            ) -> Tensor:
        memory = memory.permute(1, 0, 2) # (B, Seq[target], Features) -> (Seq[target], B, Features)
        target = target.permute(1, 0) # (B, Seq[target]) -> (Seq[target], B)

        target_emb = self.positional_encoding(self.token_emb(target)) # (Seq[target], B, Features)
        decoder_out = self.decoder(
            tgt=target_emb,
            memory=memory,
            tgt_mask=target_mask,
            memory_mask=None,
            tgt_key_padding_mask=target_padding_mask,
            ) # (Seq[target], B, Features)

        return decoder_out.permute(1, 0, 2) # (B, Seq[target], Features)

    def forward( # pylint: disable=C0116
            self,
            memory: Tensor,
            target: Tensor,
            target_mask: Tensor,
            target_padding_mask: Tensor,
            ) -> Tensor:
        out = self.forward_features(memory, target, target_mask, target_padding_mask) # (Seq[target], B, Features)
        out = self.head(out) # (B, Seq[target], vocab_size)

        return out
