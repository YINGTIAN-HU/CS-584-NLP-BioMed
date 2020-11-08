import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import *
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel, LdaMulticore
from gensim.corpora import Dictionary
from nltk.tokenize import RegexpTokenizer
from datetime import datetime
from gensim.models import KeyedVectors
import nltk
import pandas as pd
import re
import math
import spacy
import time
import os
import shutil
import gensim
import string

path1 = "/Users/yhu245/Documents/GitHub/breast_cancer_analysis/" #path saving unlabeled data
path2 = "/Users/yhu245/Downloads/DSM-language-model-1B-LARGE/" #path saving word2vec model
df_tweets = pd.read_csv(path1 + "BreastCancer_Rawdata_TWs_unlabeled.csv")

def clean_tweets(df=df_tweets, 
                 tweet_col='text' 
                 #date_col='Timestamp',
                 #start_datetime=datetime(2011,1,20, 0, 0, 0)
                ):
    
    df_copy = df.copy()
    
    # drop rows with empty values
    df_copy.dropna(inplace=True)
    
    # lower the tweets
    df_copy['preprocessed_' + tweet_col] = df_copy[tweet_col].str.lower()
    df_copy['preprocessed_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: [re.sub(r'[^A-Za-z0-9 ]+', '', word) for word in row.split()])
                                                                                                
    return df_copy
  
df_tweets_clean = clean_tweets(df_tweets)
df_tweets_clean.head()

model = gensim.models.Word2Vec(df_tweets_clean.preprocessed_text.tolist())
model.save("word2vec.model")
model = gensim.models.Word2Vec.load("word2vec.model")
model.wv.most_similar(positive=['tamoxifen'], topn=20)


word_vectors = KeyedVectors.load_word2vec_format(path2 + './trig-vectors-phrase.bin', binary=True, encoding='utf8', unicode_errors='ignore')
topn = word_vectors.most_similar(positive=['tamoxifen'], topn=10000)

import Levenshtein
item_levenshtein = []
for item in topn:
    item_name = item[0]
    levenshtein = Levenshtein.distance('tamoxifen', item_name)/max(len('tamoxifen'), len(item_name))  
    item_levenshtein.append([item_name, levenshtein])
    
sorted(item_levenshtein, key= lambda x: x[1], reverse=False)[0:100]







