import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random 
import re
import os
import unicodedata
import codecs
import itertools
from data_preprocess import pairs,voc,PAD_token,EOS_token


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_length, hidden=None):
        embedded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


rnn = nn.GRU(6,20,2)
print(rnn)
input = torch.randn(5,3,6)
print(input.shape)
print(input)
h0 = torch.randn(2,3,20)
output,h0 = rnn(input,h0)
print(output.shape)
print(output)
print(h0.shape)
print(h0)



