# import libraries
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer

import random
import string
import numpy as np
import nltk
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

# convert to string
small_talk = small_talk_responses.values()
small_talk = [str (item) for item in small_talk]


def tfidf_cosim_smalltalk(doc, query):
   query = [query]
   tf = TfidfVectorizer(use_idf=True, sublinear_tf=True)
   tf_doc = tf.fit_transform(doc)
   tf_query = tf.transform(query)
   cosineSimilarities = cosine_similarity(tf_doc,tf_query).flatten()
   related_docs_indices = cosineSimilarities.argsort()[:-2:-1]
   if (cosineSimilarities[related_docs_indices] > 0.7):
      ans = [small_talk[i] for i in related_docs_indices[:1]]
      return ans[0]

#Checking for greetings
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


#Checking for Introduce
def IntroduceMe(sentence):
    return random.choice(Introduce_Ans)


# Checking for Basic_Q_1
def basic1(sentence):
    for word in Basic_Q_1:
        if sentence.lower() == word:
            return random.choice(Basic_Ans_1)

# Checking for Basic_Q_2
def basic2(sentence):
    for word in Basic_Q_2:
        if sentence.lower() == word:
            return random.choice(Basic_Ans_2)


# Checking for Basic_Q_3
def basic3(sentence):
    for word in Basic_Q_3:
        if sentence.lower() == word:
            return random.choice(Basic_Ans_3)


# Checking for Basic_Q_34
def basic4(sentence):
    for word in Basic_Q_4:
        if sentence.lower() == word:
            return random.choice(Basic_Ans_4)

# print(all_text)
all_text = tokens

# reduce the dimensionality of text data and improve the accuracy 
def stem_tfidf(doc, query):
   query = [query]
   p_stemmer = PorterStemmer()
   
   tf = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
   stemmed_doc = [p_stemmer.stem(w) for w in doc]
   stemmed_query = [p_stemmer.stem(w) for w in query]
   
   tf_doc = tf.fit_transform(stemmed_doc)
   tf_query = tf.transform(stemmed_query)
   
   return tf_doc, tf_query

# calculates the cosine similarity between a document and a query
def cos_sim(x, y): #tf_doc, tf_query
   cosineSimilarities = cosine_similarity(x, y).flatten()
   related_docs_indices = cosineSimilarities.argsort()[:-2:-1]
   
   print(cosineSimilarities[related_docs_indices])
   
   if (cosineSimilarities[related_docs_indices] > 0.1):
      ans = [all_text[i] for i in related_docs_indices[:1]]
      
      ans = ' '.join(ans)
      return ans

   else:
      k = 'I am sorry, I cannot help you with this one. Hope to in the future.'
      return k
   
   #Generating response lemos
def response(user_response):
    x, y = stem_tfidf(all_text, user_response)
    g = cos_sim(x, y)
    print('\nPlantMedicBot: '+g)
    # print('PlantMedicBot')
    # print(g)
    return g





