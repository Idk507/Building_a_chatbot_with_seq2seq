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
from data_preprocess import pairs,voc,PAD_token,EOS_token,device,SOS_token
from EncoderRNN import EncoderRNN
from LuongAttnDecoderRNN import LuongAttnDecoderRNN
from data_prepare import batch2traindata
from train import decoder,encoder


# Define GreedySearchDecoder class
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


# Define evaluation function
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


# Define evaluation loop
def evaluateInput(encoder, decoder, searcher, voc):
    user_input = ''
    while True:
        try:
            user_input = input('> ')
            if user_input == 'q' or user_input == 'quit': break
            user_input = normalizeString(user_input)
            output_words = evaluate(encoder, decoder, searcher, voc, user_input)
            output_words = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))
        except KeyError:
            print("Error: Encountered unknown word.")

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

evaluateInput(encoder, decoder, searcher, voc)

def chat():
    # Initialize components (encoder, decoder, searcher, voc)
    encoder.eval()
    decoder.eval()
    searcher = GreedySearchDecoder(encoder, decoder)
    
    print("Chatbot: Hello! I'm your chatbot. You can start chatting or type 'q' to quit.")
    
    while True:
        try:
            user_input = input('You: ')
            if user_input.lower() == 'q' or user_input.lower() == 'quit':
                print("Chatbot: Goodbye!")
                break
            
            # Process user input
            user_input = normalizeString(user_input)
            response = evaluate(encoder, decoder, searcher, voc, user_input)
            
            # Print response
            response = ' '.join(response)
            print('Chatbot:', response)
        
        except KeyError:
            print("Chatbot: Error: Encountered unknown word.")
        except KeyboardInterrupt:
            print("Chatbot: Keyboard Interrupt. Goodbye!")
            break
        except Exception as e:
            print(f"Chatbot: Error: {str(e)}")

# Start the interactive chat
chat()
