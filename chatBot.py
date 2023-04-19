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

# add greetings

Introduce_Ans = ["My name is PlantMedicBot.", "My name is PlantMedicBot you can called me PlantMedic.", "I'm PlantMedicBot",
                 "My name is PlantMedicBot. and my nickname is PlantMedic and I am happy to solve your queries"]
GREETING_INPUTS = ("hello", "hi", "hiii", "hii", "hiiii",
                   "hiiii", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "hii there",
                      "hi there", "hello", "I am glad! You are talking to me"]

Basic_Q_1 = ("please help", "help me","please help me","can you help me")
Basic_Ans_1 = ["yes I can", "Tell me", "sure"]
Basic_Q_2 = ("okay","ok")
Basic_Ans_2 = ["Is there anything else to know?"]
Basic_Q_3 = ("no","not yet","Nop","nop")
Basic_Ans_3 = ["Okay Thank you Bye", "Have a nice day", "Bye"]
Basic_Q_4 = ("yes","yeah","Yep","yep")
Basic_Ans_4 = ["What","What else", "Tell me"]


small_talk_responses = {
'how are you': 'I am fine. Thank you for asking ',
'how are you doing': 'I am fine. Thank you for asking ',
'how do you do': 'I am great. Thanks for asking ',
'how are you holding up': 'I am fine. Thank you for asking ',
'good morning': 'Good Morning ',
'good afternoon': 'Good Afternoon ',
'good evening': 'Good Evening ',
'good day': 'Good day to you too ',
'whats up': 'The sky ',
'thank': 'Dont mention it. You are welcome ',
'thankyou': 'Dont mention it. You are welcome '
}