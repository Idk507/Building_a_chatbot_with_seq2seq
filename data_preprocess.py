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

CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

lines_filepath = "data\movie_lines.txt"
conv_filepath = "data\movie_conversations.txt"

with open(lines_filepath, 'r', encoding='latin-1') as file:
    lines = file.readlines()
for line in lines[:8]:
    print(line.strip())


line_fields = ["lineID", "characterID", "movieID", "character", "text"]
lines = {}

# Read the lines file and populate the lines dictionary
with open(lines_filepath, 'r', encoding="iso-8859-1") as f:
    for line in f:
        values = line.split("+++$+++")
        if len(values) >= len(line_fields):  # Ensure there are enough fields
            lineObj = {}
            for i, field in enumerate(line_fields):
                lineObj[field] = values[i].strip()
            lines[lineObj['lineID']] = lineObj["text"].strip()
        else:
            print(f"Skipping line due to insufficient fields: {line.strip()}")




# Debug: Print the first few entries of the lines dictionary
print("First few lines entries:")
for k in list(lines.keys())[:5]:
    print(k, lines[k])

conv_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
conversations = []

with open(conv_filepath, 'r', encoding='iso-8859-1') as f:
    for line in f:
        values = line.split("+++$+++")
        if len(values) >= len(conv_fields):  # Ensure there are enough fields
            convObj = {}
            for i, field in enumerate(conv_fields):
                convObj[field] = values[i].strip()
            
            # Convert string result from split to list
            lineIds = eval(convObj["utteranceIDs"])
            
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                lineId = lineId.strip()
                if lineId in lines:
                    convObj["lines"].append(lines[lineId])
                else:
                    print(f"Warning: Line ID {lineId} not found in lines dictionary.")
            conversations.append(convObj)
        else:
            print(f"Skipping conversation due to insufficient fields: {line.strip()}")

# Debugging output
print(f"Total conversations loaded: {len(conversations)}")
for conv in conversations[:2]:  # Print first 2 conversations for verification
    print(conv)


#extract the pairs of sentence from conversations 
qa_pairs =[]

for conversation in conversations:
    for i in range(len(conversation['lines']) - 1 ):
        input_line = conversation['lines'][i]
        target_line = conversation['lines'][i+1]
        if input_line and target_line:
            qa_pairs.append([input_line, target_line])


data_file = "formatted_movie_lines.txt"
delimiter = '\t'
delimiter = str(codecs.decode(delimiter,"unicode_escape"))

print("\n Writing newly formatted file....")
with open(data_file,'w',encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile,delimiter=delimiter)
    for pair in qa_pairs:
        writer.writerow(pair)
print("Done writing")


data_file = "formatted_movie_lines.txt"
with open(data_file,'rb') as f:
    lines = f.readlines()

for line in lines[:8]:
    print(line)

PAD_token = 0
SOS_token = 1
EOS_token = 2 

class Vocabulary:
    def __init__(self,name):
        self.name = name 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words = 3 
    
    def addSentence(self,sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word 
            self.num_words += 1
        else :
            self.word2count[word] +=1 
    
    def trim(self,min_count):
        keep_words = []
        for k,v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print(f"keep_words {len(keep_words)} / {len(self.word2index)}")
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words = 3 
        for word in keep_words:
            self.addWord(word)

    


import unicodedata

def unicodetoAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Example usage
result = unicodetoAscii("Monttreal,Francoise")
print(result)

def normalizeString(s):
    s = unicodetoAscii(s.lower().strip())
    s = re.sub(r"([.!?])",r" \1",s)
    s = re.sub(r"[^a-zA-Z.!?]+",r" ",s)
    s = re.sub(r"\s+",r" ",s).strip()
    return s

normalizeString("aa123!!s's")

data_file = "formatted_movie_lines.txt"
print("Reading and processing file ....")
lines = open(data_file,encoding='utf-8').read().strip().split('\n')
pairs = [[normalizeString(s) for s in pair.split('\t')] for pair in lines]
print("Done Reading !! ")
voc = Vocabulary("movie_vocab")


MAX_LENGTH = 10

def filterPair(p):
    return len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]




# Filter pairs with more than one element and apply the filterPairs function
pairs = [pair for pair in pairs if len(pair) > 1]
pairs = filterPairs(pairs)

for pair in pairs :
    voc.addSentence(pair[0])
    voc.addSentence(pair[1])

print("Counted words",voc.num_words)

for pair in pairs[:10]:
    print(pair)

MIN_COUNT = 3 

def trimRareWords(voc,pairs,MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        if keep_input and keep_output:
            keep_pairs.append(pair)
    print(f"Trimmed from {len(pairs)} pairs to {len(keep_pairs)} pairs")
    return keep_pairs


pairs = trimRareWords(voc,pairs,MIN_COUNT)
