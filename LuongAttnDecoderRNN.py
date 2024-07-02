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


#Luong attention layer 

class Attn(torch.nn.Module):
    def __init__(self,method,hidden_size):
        super(Attn,self).__init__()
        self.method = method 
        self.hidden_size =hidden_size

    def dot_score(self,hidden,encoder_output):
        return torch.sum(hidden * encoder_output,dim=2)
    
    def forward(self,hidden,encoder_output):
        attn_energies = self.dot_score(hidden,encoder_output)
        attn_energies = attn_energies.t() # transpose max length and batch dimensions
        return F.softmax(attn_energies,dim=1).unsqueeze(1) #return the softmax normalized probability scores 
    


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.dropout = dropout

        # layers
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):  # Corrected 'sef' to 'self'
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden


