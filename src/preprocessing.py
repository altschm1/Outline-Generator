"""
Michael Altschuler
Created: 11/18/2018
Updated: 11/18/2018

preprocessing.py
Perform NLP preprocessing techniques such as:
    paragraph/sentence/word tokenization
    capitalization/punctuation-removal
    lemmatization
    vocabulary extraction
    paragraph/sentence/word positioning

Pre-compilation:
pip install nltk
"""

import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

def download():
    """downloads all needed files/datasets from nltk"""
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

def tokenize_to_paragraphs(text):
    """takes in a body of text and splits it up by paragraphs"""
    return text.split("\n")

def tokenize_to_sentences(text):
    """takes in a body of text and splits it up by sentences"""
    return nltk.sent_tokenize(text)

def tokenize_to_words(text):
    """takes in a body of text and splits it up by words"""
    return nltk.word_tokenize(text)

def remove_capitalization(text):
    """takes in a body of text and returns the same text without capitalization"""
    return text.lower()

def remove_punctuation(text):
    """this takes in a body of text and returns the same text without punctuations"""
    output = ""

    for char in text:
        if char not in string.punctuation:
            output = output + char

    return output

def part_of_speech_tagger(text):
    return nltk.pos_tag(tokenize_to_words(text))

def remove_stop_words(text):
    """takes in a body of text, and returns the list without stopwords as defined in nltk"""
    
    words = tokenize_to_words(text)
    stop_words = set(stopwords.words('english'))
    
    output = []

    for word in words:
        if word not in stop_words:
            output.append(word)

    return output

def get_pos_lemm(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def remove_blank_space(text):
    output = ""
    for char in text:
        if char not in string.whitespace:
            output = output + char

    return output


def lemmatize(text):
    """returns the lemmatization of  each word in text"""
    words = part_of_speech_tagger(text)
    lemmatizer = WordNetLemmatizer()
    output = []
    
    for word in words:
        tag = get_pos_lemm(word[1])
        output.append(lemmatizer.lemmatize(word[0], tag))

    return output

def extract_vocab(text):
    text = remove_punctuation(text)
    text = remove_capitalization(text)

    words = set(tokenize_to_words(text))
    vocab = set(remove_stop_words(text))
    return list(vocab)

     

if __name__ == "__main__":
    text = \
    """Hello! My name is Michael. I go to TCNJ.
I am building a class that takes performs some basic NLP operations \
on it's field called text.  Some operations include tokenization, stopword removal, lemmatization, \
punctuation removal, and capitalization removal.
The main method is a test case."""
    print(text)
    print(tokenize_to_paragraphs(text))
    print(tokenize_to_sentences(text))
    print(tokenize_to_words(text))
    print(remove_capitalization(text))
    print(remove_punctuation(text))
    print(remove_stop_words(text))
    print(part_of_speech_tagger(text))
    print(lemmatize(text))
    print(extract_vocab(text))