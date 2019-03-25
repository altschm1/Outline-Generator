"""
Michael Altschuler
Created: 11/18/2018
Updated: 3/9/2019

preprocessing.py
Perform NLP preprocessing techniques such as:
    paragraph/sentence/word tokenization
    capitalization/punctuation-removal
    lemmatization
    vocabulary extraction
    part of speech tagging

Pre-compilation (make sure to have the following downloaded):
pip install nltk
python
>> import preprocessing
>> preprocessing.download() # downloads all the necessary files/packages from nltk
"""

import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

def download():
    """downloads all needed files/datasets from nltk
    
    Preconditions:
    none
    Postconditions:
    punkt, stopwords, averaged_perceptron_tagger, and wordnet from nltk will be installed on local machine
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

def tokenize_to_paragraphs(text):
    """takes in a body of text and splits it up by paragraphs
    
    Preconditions:
    text -- string -- the body of text to be tokenized
    Postconditions:
    returns list -- a list of paragraphs
    """
    return text.split("\n")

def tokenize_to_sentences(text):
    """takes in a body of text and splits it up by sentences
    
    Preconditions:
    text -- string -- the body of text to be tokenized
    Postconditions:
    returns list -- a list of sentences
    """
    return nltk.sent_tokenize(text)

def tokenize_to_words(text):
    """takes in a body of text and splits it up by words
    
    Preconditions:
    text -- string -- the body of text to be tokenized
    Postconditions:
    returns list -- a list of words
    """
    return nltk.word_tokenize(text)

def remove_capitalization(text):
    """takes in a body of text and returns the same text without capitalization
    
    Preconditions:
    text -- string -- the body of text to be modified
    Postconditions:
    returns string -- the parameter as written in lowercase
    """
    return text.lower()

def remove_punctuation(text):
    """this takes in a body of text and returns the same text without punctuations
    
    Preconditions:
    text -- string -- the body of text to be modified
    Postconditions:
    returns string -- the text with all punctuation removed
    """
        
    output = ""

    for char in text:
        if char not in string.punctuation:
            output = output + char

    return output

def part_of_speech_tagger(text):
    """this takes a body of text and returns the part of speech for each word
    
    Preconditions:
    text -- string -- the body of text to parse
    Postconditions:
    returns list -- a list of tuples where the first element of the tuple is the word and the second tuple is POS
    """
    return nltk.pos_tag(tokenize_to_words(text))

def remove_stop_words(text):
    """takes in a body of text, and returns the list without stopwords as defined in nltk
    
    Preconditions:
    text -- string -- the body of text to parse
    Postconditions:
    returns list -- a list of words from text that does not belong to nltk.stop_words
    """
    
    words = tokenize_to_words(text)
    stop_words = set(stopwords.words('english'))
    
    output = []

    for word in words:
        if word not in stop_words:
            output.append(word)

    return output

def get_pos_lemm(treebank_tag):
    """get the part-of-speech given the output from nltk.pos_tag()
    
    Precondition:
    treebank_tag -- string -- 2nd element of tuple of nltk.pos_tag()
    Postconditions:
    returns wordnet.attribute -- the proper wordnet tag as needed for lemmatization
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def remove_blank_space(text):
    """this takes in a body of text and returns the same text without blank spaces
    
    Preconditions:
    text -- string -- the body of text to be modified
    Postconditions:
    returns string -- the text with all blank spaces removed
    """
    output = ""
    for char in text:
        if char not in string.whitespace:
            output = output + char

    return output


def lemmatize(text):
    """returns the lemmatization of  each word in text
    
    Precondition:
    text -- string -- the body of text to be lemmatized
    Postconditions:
    returns list -- list of words in lemmatized form
    """
    words = part_of_speech_tagger(text)
    lemmatizer = WordNetLemmatizer()
    output = []
    
    for word in words:
        tag = get_pos_lemm(word[1])
        output.append(lemmatizer.lemmatize(word[0], tag))

    return output

def extract_vocab(text):
    """returns a list of words that encompass the entire vocabulary from the body of text
    
    Precondition:
    text -- string -- the body of text to be parsed
    Postconditions:
    returns list -- list of words that don't include nltk.stop_words
    """
    text = remove_punctuation(text)
    text = remove_capitalization(text)

    words = set(tokenize_to_words(text))
    vocab = set(remove_stop_words(text))
    return list(vocab)

def remove_duplicate_sentences(sentences):
    """removes all duplicate sentences
    
    Precondition:
    sentences -- list -- a list of sentences
    Postconditions:
    returns list -- list of non-duplicate sentences
    """
    return list(set(sentences))

     
# For testing purposes
#FIXME: REPLACE WITH PYTHON RECOMMENDED UNIT TESTS
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