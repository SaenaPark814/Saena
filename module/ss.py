# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch import nn
import torch
from util import get_device
import numpy as np

from module.positional_embedding import Embeddings
from util.tokens import PAD_TOKEN_ID, SOS_TOKEN_ID


device = get_device()

class Transformer(nn.Module):
    def __init__(self, encoder_params, decoder_params) :

        super(Transformer, self).__init__()
        d_model = encoder_params.embedding_dim
        n_head = encoder_params.n_head
        num_encoder_layers = encoder_params.num_encoder_layer
        num_decoder_layers = encoder_params.num_decoder_layer
        dim_feedforward = encoder_params.dim_feedforward
        dropout = encoder_params.dropout_prob

        self.src_embedding = Embeddings(params = encoder_params)
        self.tgt_embedding = Embeddings(params = decoder_params)

        self.transfomrer = nn.Transformer(d_model=d_model,
                             nhead=n_head,
                            # num_encoder_layers=num_encoder_layers,
                            # num_decoder_layers=num_decoder_layers,
                            # dim_feedforward=dim_feedforward,
                             dropout=dropout)

        self.proj_vocab_layer = nn.Linear(in_features=d_model, out_features=decoder_params.vocab_size)
        self.apply(self._initailze)

    def forward(self, enc_input, enc_lengths, dec_input, dec_lengths) :
        x_enc_embed = self.src_embedding(enc_input.long())
        x_dec_embed = self.tgt_embedding(dec_input.long())

        # Masking
        src_key_padding_mask = enc_input == PAD_TOKEN_ID
        tgt_key_padding_mask = dec_input == PAD_TOKEN_ID
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = self.transfomrer.generate_square_subsequent_mask(dec_input.size(1))

        x_enc_embed = torch.einsum('ijk->jik', x_enc_embed)
        x_dec_embed = torch.einsum('ijk->jik', x_dec_embed)

        feature = self.transfomrer(src = x_enc_embed,
                                   tgt = x_dec_embed,
                                   src_key_padding_mask = src_key_padding_mask,
                                  # tgt_key_padding_mask = tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   tgt_mask = tgt_mask.to(device))
        logits = self.proj_vocab_layer(feature)
        logits = torch.einsum('ijk->jik', logits)


        batch_size = x_enc_embed.size(1)
        # (Batch_size)
        initial_input = batch_size * [SOS_TOKEN_ID]
        initial_input = torch.tensor(initial_input, dtype=torch.long, device=device).unsqueeze(
            -1)

        decoder_input = initial_input

        predictions = []
        for t in range(self.max_seq_len):

            if self.training:
                # teacher forcing
                decoder_input = dec_input[:, t]
            else:
                # Greedy search
                top_value, top_index = logits.data.topk(1)
                decoder_input = top_index.squeeze(-1).detach()
                predictions.append(decoder_input.cpu())

            decoder_input = decoder_input.long().unsqueeze(-1)

        return logits, predictions

    def _initailze(self, layer):
        if isinstance(layer, (nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight)

    # TODO: check the code
    def init_src_embedding_weight(self,
                              weight: np.ndarray):
        self.src_embedding.weight = nn.Parameter(torch.from_numpy(weight), requires_grad=False)

    def init_tgt_embedding_weight(self,
                              weight: np.ndarray):
        self.tgt_embedding.weight = nn.Parameter(torch.from_numpy(weight), requires_grad=False)