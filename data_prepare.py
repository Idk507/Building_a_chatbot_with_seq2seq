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

def indexesfromsentence(voc,sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

inp = []
out = []
i = 0
for pair in pairs[:10]:
    inp.append(pair[0])
    out.append(pair[1])

print(inp)
print(len(inp))

indexes = [indexesfromsentence(voc,sentence) for sentence in inp]

def zeropadding(l,fillvalue=0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

leng = [len(ind) for ind in  indexes]
max(leng)

test_result = zeropadding(indexes)
print(len(test_result))

for i,seq in enumerate(test_result):
    print(i,seq)
    #for tok in seq:
        #print(tok)


def binarymatrix(l,value =0):
    m = []
    for i,seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

binary_result = binarymatrix(test_result)

def inputvar(l,voc):
    indexes_batch = [indexesfromsentence(voc,sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeropadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar,lengths

def outputVar(l,voc):
    indexes_batch = [indexesfromsentence(voc,sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeropadding(indexes_batch)
    mask = binarymatrix(padList)
    mask = torch.ByteTensor(mask)
    padvar = torch.LongTensor(padList)
    return padvar,mask,max_target_len

def batch2traindata(voc,pair_batch):
    pair_batch.sort(key=lambda x : len(x[0].split(" ")),reverse = True)
    input_batch,output_batch = [],[]
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp,lengths = inputvar(input_batch,voc)
    output,mask,max_target_len = outputVar(output_batch,voc)
    return inp,lengths,output,mask,max_target_len


#validation example 

small_batch =5 
batches = batch2traindata(voc,[random.choice(pairs)for _ in range(small_batch)])
input_variable,lengths,target_variable,mask,max_target_len = batches

print("input_variable",input_variable)
print("lengths",lengths)
print("target_variable",target_variable)
print("mask",mask)
print("max_target_len",max_target_len)


