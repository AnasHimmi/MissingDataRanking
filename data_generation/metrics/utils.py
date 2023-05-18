#!/usr/bin/python2
# -*- coding: utf-8 -*-
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import json

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")
stopset = frozenset(stopwords.words('english'))

def stem_word(word):
    return stemmer.stem(normalize_word(word))

def normalize_word(word):
    return word.lower()

def get_len(element):
    return len(tokenizer.tokenize(element))

def sentence_tokenizer(sentence):
    return tokenizer.tokenize(sentence)

def get_ngrams(sentence, N):
    tokens = tokenizer.tokenize(sentence.lower())
    clean = [stemmer.stem(token) for token in tokens]
    return [gram for gram in ngrams(clean, N)]

def get_words(sentence, stem=True):
    if stem:
        words = [stemmer.stem(r) for r in tokenizer.tokenize(sentence)]
        return [normalize_word(w) for w in words]
    else:
        return [normalize_word(w) for w in tokenizer.tokenize(sentence)]
    
def tokenize(text):
#    return [w.lower() for sent in text for w in tokenizer.tokenize(sent) if w not in stopset]
    return [w.lower() for sent in text for w in tokenizer.tokenize(sent)]

def load_json(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))  
    return data

def load_text(path):
    lines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            l = line.strip()
            lines.append(l)
    return lines