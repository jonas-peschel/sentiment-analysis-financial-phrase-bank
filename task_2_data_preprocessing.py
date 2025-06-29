import numpy as np
import string 
import re 
import spacy
spacy.cli.download("en_core_web_sm") 
nlp = spacy.load("en_core_web_sm") 

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords 
from nltk import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from gensim import downloader

def load_data(config, file_path):

    texts, labels = [], [] 
    with open(file_path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            text, label = line.strip().split("@")

            if not (config.only_binary & (label == "neutral")):

                texts.append(text)
                labels.append(label)

    LABELS = np.unique(labels).tolist() # unique labels
    labels_num = [LABELS.index(label) for label in labels]

    return texts, labels, labels_num, LABELS 

def pretrained_dense_embeddings(texts, model):
    """
    Calculate dense sentence embeddings using pre-trained word
    embedding model and averaging of the word embeddings.
    """

    X = []

    for text in texts:

        words = word_tokenize(text)
        word_embeddings = []

        for word in words:
            try:
                word_embeddings.append(model[word])
            except:
                continue

        # averaging
        if word_embeddings:
            X.append(np.mean(np.array(word_embeddings), axis=0))
        else:
            X.append(np.zeros(model.vector_size))

    return np.array(X)

def data_preprocessing(config, data_filepath):

    texts, _, labels, LABELS = load_data(config, data_filepath)

    ## pre-processing steps
    texts = [text.lower() for text in texts] # always lowercase

    if config.remove_punct:
        texts = ["".join([char for char in text if char not in string.punctuation]) for text in texts]

    if config.remove_stopwords:
        stop_words = stopwords.words("english")
        texts = [" ".join([word for word in word_tokenize(text) if word not in stop_words]) for text in texts]

    if config.remove_nums:
        texts = [re.sub(r"\d+", "", text) for text in texts]

    if config.lemmatize:
        docs = [nlp(text) for text in texts]
        texts = [" ".join([token.lemma_ for token in doc]) for doc in docs]

    # remove whitespaces
    texts = [text.strip() for text in texts]

    ## text vectorization
    if config.vectorization == "BagOfWords":

        BoW_Vectorizer = CountVectorizer(ngram_range=(1,1))
        X = BoW_Vectorizer.fit_transform(texts).toarray()

    elif config.vectorization == "TF-IDF":

        Tfidf_Vectorizer = TfidfVectorizer(ngram_range=(1,1))
        X = Tfidf_Vectorizer.fit_transform(texts).toarray()

    elif config.vectorization == "Glove-50":

        model = downloader.load("glove-wiki-gigaword-50")
        X = pretrained_dense_embeddings(texts, model)

    elif config.vectorization == "Glove-200":

        model = downloader.load("glove-wiki-gigaword-200")
        X = pretrained_dense_embeddings(texts, model)

    else: 

        X = texts


    y = np.array(labels)
  
    return X, y, LABELS



