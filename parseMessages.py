from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
import keras.utils as ku

import pickle
import pandas as pd
import numpy as np

import string, json, os, math
from os import listdir

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
total_num_samples = 0  # Number of samples to train on.
inputText = []
outputText = []
path = "./messages"

def cleanText(text):
    text = text.replace("-", " ")
    text = "".join(v for v in text if v not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii",'ignore')
    return text

def decode(jsonFile):
    global total_num_samples
    f = open(jsonFile,)
    link_dict = json.loads(f.read())
    allMess = link_dict["messages"]
    allMessLen = len(allMess) - 1
    while allMessLen > 0:
        buildq = ""
        if allMess[allMessLen]["sender_name"] != "Austin Cheng" and "content" in allMess[allMessLen]:
            name = allMess[allMessLen]["sender_name"]
            while allMessLen > 0 and name == allMess[allMessLen]["sender_name"]:
                if "content" in allMess[allMessLen]:
                    buildq = buildq + " " + allMess[allMessLen]["content"]
                allMessLen -= 1
            buildr = ""
            if allMess[allMessLen]["sender_name"] == "Austin Cheng" and "content" in allMess[allMessLen]:
                while allMessLen > 0 and allMess[allMessLen]["sender_name"] == "Austin Cheng":
                    if "content" in allMess[allMessLen]:
                        buildr = buildr + " " + allMess[allMessLen]["content"]
                    allMessLen -= 1
                #print("buildr: " + buildr)
                holdr = cleanText(buildr)
                #print("buildq:  " + buildq)
                holdq = cleanText(buildq)
                if (len(holdr) != 0 and len(holdq) != 0):
                    inputText.append("BOS" + holdq + "EOS")
                    print("holdq: " + holdq)
                    outputText.append("BOS" + holdr + "EOS")
                    print("holdr: " + holdr)
                    total_num_samples += 1
                    print(total_num_samples)
        allMessLen -= 1

for folder in listdir(path):
    holdpath = path + '/' + folder
    for folder2 in listdir(holdpath):
        holdpath2 = holdpath + '/' + folder2
        for isJson in listdir(holdpath2):
            if isJson[-5:] == '.json':
                decode(holdpath2 + '/' + isJson)
train_samples = math.floor(total_num_samples / 10 * 7)
train_input = inputText[:7000]
train_output = outputText[:7000]
tokenizer = Tokenizer()
