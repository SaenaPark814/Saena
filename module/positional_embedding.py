# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#import os

#import numpy as np
import torch
#from torch import nn
#from torch.utils.data import DataLoader
#from tqdm import tqdm

import torch.nn as nn
import math
from util.tokens import PAD_TOKEN_ID


class PositionalEmbedding(nn.Module):
    """
    ref: https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/embedding/position.py
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]



# class TokenEmbedding(nn.Module):
#     def __init__(self, vocab_size, embed_size, pad_id):
#         super(TokenEmbedding, self).__init__()
#         self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
#
#     def forward(self, x):
#         x_embed = self.token_embedding(x)
#         return x_embed



class Embeddings(nn.Module):
    def __init__(self, params):
        super(Embeddings, self).__init__()
        # self.token_embedding = TokenEmbedding(vocab_size=encoder_params.vocab_size, embed_size=encoder_params.embedding_dim,
        #                                       #pad_id=vocab.token2idx[vocab.PAD]
        #                                       pad_id = 0)

        self.token_embedding = nn.Embedding(params.vocab_size,  #num_embedding
                                            params.embedding_dim,  # embedding_dim
                                            PAD_TOKEN_ID   # padding_idx
                                            )
        self.pos_embedding = PositionalEmbedding(d_model = params.embedding_dim, max_len= params.max_seq_len)

    def forward(self, x):
        token_embed = self.token_embedding(x)
        pos_embed = self.pos_embedding(x)
        return token_embed + pos_embed