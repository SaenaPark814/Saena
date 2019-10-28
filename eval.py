# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import nltk
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ParallelTextDataSet
from module import Seq2Seq
from module import GruEncoder, GruDecoder
from module.transformer import Transformer
from module.tokenizer import MecabTokenizer
from module.tokenizer import NltkTokenizer
from params import decoder_params
from params import encoder_params
from params import eval_params
from util import AttributeDict
from util import eval_step
from util import get_device
from util.tokens import PAD_TOKEN_ID


def check_params(config: AttributeDict):
    assert config.get('src_tokenizer', '') in [
        MecabTokenizer, NltkTokenizer
    ], 'src_tokenizer should be one of following [MecabTokenizer, NltkTokenizer]'
    assert config.get('tgt_tokenizer', '') in [
        MecabTokenizer, NltkTokenizer
    ], 'tgt_tokenizer should be one of following [MecabTokenizer, NltkTokenizer]'
    assert config.get('src_vocab_filename', None) is not None, \
        'src_vocab_filename must not be None'
    assert config.get('tgt_vocab_filename', None) is not None, \
        'tgt_vocab_filename must not be None'
    assert config.get('src_word_embedding_filename', None) is not None, \
        'src_word_embedding_filename must not be None'
    assert config.get('tgt_word_embedding_filename', None) is not None, \
        'tgt_word_embedding_filename must not be None'
    assert config.get('src_corpus_filename', None) is not None, \
        'src_corpus_filename must not be None'
    assert config.get('tgt_corpus_filename', None) is not None, \
        'tgt_corpus_filename must not be None'
    assert config.get('encoder', None) is not None, \
        'encoder should not be None'
    assert config.get('decoder', None) is not None, \
        'decoder should not be None'
    assert config.get('checkpoint_path', None) is not None, \
        'model_path should not be None'


def check_vocab_embedding(
        vocab_file_path: str,
        word_embedding_file_path: str,
):
    """
    :return: word2id, id2word, embedding_matrix
    """

    with open(vocab_file_path, mode='r', encoding='utf-8') as f:
        tokens = f.readlines()
    word2id = {}
    id2word = {}
    for index, token in enumerate(tokens):
        token = token.strip()
        if len(token) == 0:
            continue
        word2id[token] = index
        id2word[index] = token

    embedding_matrix = np.load(word_embedding_file_path)

    return word2id, id2word, embedding_matrix


def eval_model(model: nn.Module,
               loss_func,
               test_data_loader: DataLoader,
               device: str,
               id2word: dict):
    model.eval()

    with torch.no_grad():
        losses = []
        data_length = len(test_data_loader)

        predictions = []
        target_sequences = []

        with tqdm(test_data_loader, total=data_length, desc=f'EVAL') as tqdm_iterator:
            for _, batch in enumerate(tqdm_iterator):
                _, _, tgt_seqs, tgt_lengths = batch

                # TODO: PAD 고려??
                loss, logits, preds = eval_step(model, device, batch, loss_func)
                preds = torch.cat(preds).view(-1, len(preds))

                for pred in preds:
                    # idx list -> word list
                    sentence = []
                    for token in pred:
                        token = token.item()
                        if token == PAD_TOKEN_ID:
                            break
                        sentence.append(id2word[token].strip())
                    predictions.append(sentence)

                for tgt_seq in tgt_seqs:
                    sentence = []
                    for token in tgt_seq:
                        token = token.item()
                        if token == PAD_TOKEN_ID:
                            break
                        sentence.append(id2word[token].strip())
                    target_sequences.append(sentence)

                losses.append(loss)
                tqdm_iterator.set_postfix_str(f'loss: {loss:05.3f}')

        bleu_score = nltk.translate.bleu_score.corpus_bleu(target_sequences, predictions)
        print(f'BLEU score: {bleu_score}')

    avg_loss = np.mean(losses)
    print(f'EVAL avg losses: {avg_loss:05.3f}')

    return avg_loss