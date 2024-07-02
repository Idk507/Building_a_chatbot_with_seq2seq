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
from data_preprocess import pairs,voc,PAD_token,EOS_token,device
from EncoderRNN import EncoderRNN
from LuongAttnDecoderRNN import LuongAttnDecoderRNN
from data_prepare import batch2traindata


#https://medium.com/@mbednarski/understanding-indexing-with-pytorch-gather-33717a84ebc4
    
def maskNLLoss(decoder_output, target, mask):
    nTotal = mask.sum()  # how many elements should be considered
    target = target.view(-1, 1)
    gathered_output = torch.gather(decoder_output, 1, target)
    crossEntropy = -torch.log(gathered_output)
    loss = crossEntropy.masked_select(mask.bool())  # convert mask to bool
    loss = loss.mean()
    loss = loss.to(device)
    return loss, nTotal.item()


#visualizing 

small_batch_size = 5
batches = batch2traindata(voc,[random.choice(pairs) for _ in range(small_batch_size)])

input_variable,lengths,target_variable,mask,max_target_len = batches

print("input_variable",input_variable)

print("lengths",lengths)

print("target_variable",target_variable)

print("mask",mask)

print("max_target_len",max_target_len)


#define the parameters 

hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
attn_model = 'dot'
embedding = nn.Embedding(voc.num_words,hidden_size) #no of words ,dimension of the mbeddings that basically we are representing the value

# Correct the order of parameters in EncoderRNN initialization
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

encoder = encoder.to(device)
decoder = decoder.to(device)

print(encoder)
print(decoder)

#ensure dropout layers to train 

encoder.train()
decoder.train()

#initialize optimizer 
encoder_optimizer = optim.Adam(encoder.parameters(),lr=0.0001)
decoder_optimizer = optim.Adam(decoder.parameters(),lr=0.001)
encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()

input_variable = input_variable.to(device)
lengths = lengths.to('cpu')

target_variable = target_variable.to(device)
mask = mask.to(device)


loss = 0
print_losses = []

n_totals = 0

encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
print("Encoder Outputs Shape:", encoder_outputs.shape)
print("Last Encoder Hidden Shape:", encoder_hidden.shape)

decoder_input = torch.LongTensor([[SOS_token for _ in range(small_batch_size)]])

decoder_input = decoder_input.to(device)

print("Decoder Input Shape:", decoder_input.shape)
print(decoder_input)


#set intial decoder hidden states 


decoder_hidden = encoder_hidden[:decoder.n_layers]

print("Decoder Hidden Shape:", decoder_hidden.shape)


# ASsume we are using teacher forcing 

for t in range(max_target_len):
    decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden,encoder_outputs)
    print("Decoder Output shape :",decoder_output.shape)
    print("Decoder Hidden State", decoder_hidden.shape)

    decoder_input = target_variable[t].view(1,-1)
    print("The target variable at the current timestep before reshaping", target_variable[t])
    print("target variable at current timestep shape before reshaping ",target_variable[t].shape)
    print("The target variable at the current timestep after reshaping", target_variable[t].view(1,-1))
    print("target variable at current timestep shape after reshaping ",target_variable[t].view(1,-1).shape)
    print("the decoder input shape (reshape the target variable ) ",decoder_input.shape)

    #calculate and accumulate loss 
    print("The mask at the current timestemp ",mask[t])
    print("The mask at the curren timestep shape ",mask[t].shape)
    mask_loss,nTotal = maskNLLoss(decoder_output,target_variable[t],mask[t])
    print("The mask loss ",mask_loss)
    print("The mask loss shape ",mask_loss.shape)
    print("The nTotal ",nTotal)
    loss += mask_loss
    print_losses.append(mask_loss.item() * nTotal)
    print("The print losses ",print_losses)
    n_totals += nTotal
    print("The n_totals ",n_totals)
    encoder_optimizer.step()
    decoder_optimizer.step()
    returned_loss = sum(print_losses)/n_totals
    print("The returned loss ",returned_loss)
 
    print("\n")
    print("----------------------------------done for one timestep-------------------------------")
    print("\n")
    

def train(input_variable,lengths,target_variable,mask,max_target_len,encoder,decoder,embedding,encoder_optimizer,decoder_optimizer,batch_size,clip,max_length=MAX_LENGTH):
    #zero gradients 
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #set device options 

    input_variable = input_variable.to(device)
    lengths = lengths.to('cpu')
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    #intizlize variables 
    loss = 0
    print_losses = []
    n_totals = 0

    #forward pass through encoder 
    encoder_outputs,encoder_hidden = encoder(input_variable,lengths)

    #create initial decoder hidden state to encoder final hidden state 
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    teacher_forcing_ratio = 0.5
    #determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing : 
        for t in range(max_target_len):
            decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden,encoder_outputs)
            decoder_input = target_variable[t].view(1,-1)
            mask_loss,nTotal = maskNLLoss(decoder_output,target_variable[t],mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else :
        for t in range(max_target_len):
            decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden,encoder_outputs)
            _,topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            mask_loss,nTotal = maskNLLoss(decoder_output,target_variable[t],mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(),clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(),clip)

    #adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)



# Define the directory to save checkpoints
save_dir = 'checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
clip = 50.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

embedding = nn.Embedding(voc.num_words, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN('dot', embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

encoder = encoder.to(device)
decoder = decoder.to(device)

encoder.train()
decoder.train()

print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

for iteration in range(1, n_iteration + 1):
    training_batch = [random.choice(pairs) for _ in range(batch_size)]
    input_variable, lengths, target_variable, mask, max_target_len = batch2traindata(voc, training_batch)
    loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
    print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, loss))
    if iteration % save_every == 0:
        torch.save({
            'iteration': iteration,
            'en': encoder.state_dict(),
            'de': decoder.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': loss,
            'voc_dict': voc.__dict__,
            'embedding': embedding.state_dict()
        }, os.path.join(save_dir, '{}_{}.tar'.format(iteration, 'checkpoint')))


