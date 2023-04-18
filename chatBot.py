# import libraries
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer

import random
import string
import numpy as np
import nltk
import datetime
from nltk.chat.util import Chat, reflections 

# * first-time use only
# nltk.download('punkt')
# nltk.download('wordnet')

import warnings
warnings.filterwarnings("ignore")

# open dataset
ds = open("dataset.txt", 'r', errors= "ignore")

# normalization

raw = ds.read() # separated sections file
lraw = raw.lower() # convert to lower case

# print(raw)
# print(lraw)

tokens = nltk.sent_tokenize(lraw) # convert to list of sentences

# print(tokens)

# lemmartization

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return[lemmer.lemmatize(token)for token in tokens]

remove_punct_dict = dict((ord(punct), None)for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
