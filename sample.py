# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:09:16 2020

@author: Nairuti
"""
#import pandas as pd
from keras.preprocessing.text import Tokenizer
import numpy as np
import re
from keras.utils import to_categorical
#import regex
#import time

#reading the data from csv
#d=open('sample.txt')
d=open('sample2.txt',errors='ignore')
data = d.readlines()

#removing the redundant lines


# Method that will clean the data:
def clean_text(text):
    text = text.lower() #convert all the chracters into small letters
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"n't", "not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}'+=|.!?,]", "", text)
    text = text.replace("[", "")
    text = text.replace("]", "")
    return text

sentence_count = {}
total_sentences = 0
for line in data:
    if line not in sentence_count:
        sentence_count[line] = 1
    else:
        sentence_count[line] += 1
    total_sentences += 1
#print(sentence_count)
      
# Removing the lines that contains numeric data 
unique_data_str = []
for i in range(len(data)):
    if type(data[i]) is str:
        unique_data_str.append(data[i])
    else:
        None
print()
print()
print()
#print(unique_data_str)
# Cleaning the data
clean_data = []
for text in unique_data_str:
    a = re.sub(r'[^a-zA-z ]+', '', text).strip()
    if len(a)>0:
        clean_data.append(clean_text(a))
    else:
        None
print()
print()
print()
#print(clean_data)
# Removing the lines which are to short or to long


# Counting the appearnce of each word in the corpus also calculates the number of unique words also

word2count = {}
total_words = 0
for text in clean_data:
    for word in text.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
        total_words += 1
#print(word2count)        
# creating a list that will only contain the words that appear more than 15 times

# Total number of words in corpus after removing the words which appears less than 15 times and further cleaning
total_words_d15 = 0
for line in clean_data:
    for word in line.split():
      total_words_d15 += 1 
#print(total_words_d15)
""" Initially we had 437579 lines in our data after cleaning and preprocessing the data
    now our complete data will have 126003 lines, that means we removed 71.20% data which was useless"""   
    
#defining a function to save data
def write_txt(name, data):
    file1 = open("{0}.txt".format(name),"w") 
    for line in data:
        file1.writelines(line) 
        file1.writelines('\n') 
    file1.close() #to change file access modes

# Reading text file
#fl = open("EU-AU-Description-19-9-2019.txt","r+")  
#clean_unique_data = fl.read().splitlines()


# writing data to text files

write_txt(name = 'sample3', data = clean_data)

def read_file(filepath):
	with open(filepath) as f:
		str_text = f.read()
	return str_text
import json
text = read_file('sample3.txt')
tokens = text.split(" ")
tokens.pop(0)
#print(tokens)
train_len = 3+1
with open('text_sequences.json') as f:
    text_sequences=json.load(f)
#text_sequences = []
for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)
    
print(text_sequences)
with open('text_sequences.json','w') as fp:
    json.dump(text_sequences,fp)

with open('data.json') as f:
  data = json.load(f)
for key in word2count:
    
    
    if key in data:
        
        data[key]=data[key]+word2count[key]
        
for key in list(word2count):
    if key in data:
        del word2count[key]
        #del word2count[key]


data.update(word2count)        
data=sorted(data.items(), key=lambda x: x[1], reverse=True)
data=dict(data)
#print(data)
with open('data.json', 'w') as fp:
    #json.dump(tokenizer.word_counts, fp)
    json.dump(data,fp)
sequences = {}
count = 1
for i in range(len(tokens)):
    if tokens[i] not in sequences:
        
        sequences[tokens[i]] = count
        
        count += 1
#print(sequences)
print()
print()

#print(sequences)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(text_sequences)

sequences = tokenizer.texts_to_sequences(text_sequences) 

#Collecting some information   
unique_words = tokenizer.index_word
#unique_words=data.index_word
#print(unique_words)
unique_wordsApp = tokenizer.word_counts
vocabulary_size = len(tokenizer.word_counts)
#vocabulary_size=len(data)
#print(vocabulary_size)
#print(tokenizer.word_counts)

#print(sequences)
n_sequences = np.empty([len(sequences),train_len], dtype='int32')
for i in range(len(sequences)):
    n_sequences[i] = sequences[i]

#print(n_sequences)

train_inputs = n_sequences[:,:-1]
#print(train_inputs)

#ntraininp=np.append(train_inputs,ntraininp)

train_targets = n_sequences[:,-1]
#ntraintar=np.append(train_targets,ntraintar)
#print(train_targets)
train_targets = to_categorical(train_targets, num_classes=vocabulary_size+1)
seq_len = train_inputs.shape[1]
train_inputs.shape

#print(seq_len)
#print(vocabulary_size)


#print(data)
