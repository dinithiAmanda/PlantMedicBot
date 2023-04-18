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

raw = ds.read() # separated sections file
lraw = raw.lower() # convert to lower case

# print(raw)
# print(lraw)

tokens = nltk.sent_tokenize(lraw) # convert to list of sentences

print(tokens)


