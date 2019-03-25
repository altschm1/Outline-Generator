"""
Michael Altschuler
Created: 11/18/2018
Updated: 3/9/2019

vector.py
Convert preprocessed input into the proper vector representation (2D numpy array)
Currently implemented functions:
    convert_bow(sentences, vocab) --> converts sentence into bag of words vectors
    convert_doc2vec(sentences) --> converts each sentence into doc2vec sentence embedding
Not implemented yet:
    convert_tfidf()
    convert_infersent()
    convert_skip_thought_embeddings()
    convert_pretrained_doc2vec()

Pre-compilation:
pip install numpy
pip install nltk
pip install gensim
"""

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def _convert_bag_word_vector(sentence, vocab):
    """convert each sentence to a bag of word vector
    where each component is a word from the vocab and the element is the count of the word
    
    Preconditions: 
    sentence -- string -- a sentence to be represented as a single vector (all lowercase)
    vocab -- list -- the vocabulary from the corpus (all lowercase)
    Postconditions:
    returns numpy array shape(1, len(vocab)) -- sentence embedding using bow
    """
    result = np.zeros(len(vocab), dtype = float)
    
    sorted_vocab = sorted(vocab)
    
    for x in range(len(sorted_vocab)):
        for s in sentence:
            if sorted_vocab[x] == s:
                result[x] = result[x] + 1.0
    return result

def convert_bow(sentences, vocab):
    """convert list of sentences into bag of word vectors
    
    Preconditions: 
    sentences -- list -- a list of sentence to be vectorized
    vocab -- list -- the vocabulary from the corpus (all lowercase)
    Postconditions:
    returns numpy array shape(2, len(vocab)) -- sentence embeddings using bow
    """
    result = np.zeros(dtype = float, shape=(len(sentences), len(vocab)))
    
    for i, s in enumerate(sentences):
        result[i] = _convert_bag_word_vector(word_tokenize(s.lower()), vocab)
    
    return result

def convert_skip_thought_embeddings():
    """convert to skip thoughts based off https://github.com/ryankiros/skip-thoughts"""
    #FIXME: COULD NOT SYNC UP WITH PYTHON3 REACHED critical error
    pass

def convert_doc2vec(sentences, max_epochs = 100, vec_size = 100, alpha = 0.05, model = 0):
    """convert to doc2vec for sentences embeddings
    
    Preconditions:
    sentences -- list -- a list of sentences to be vectorized
    max_epochs -- int -- the number of iterations/passes of the training set -- default is 100
    vec_size -- int -- the number of elements of the sentence embedding -- default is 100
    alpha -- float -- the initial learning rate and minimum learning rate
    model -- int (0 or 1) -- if 0, distributed memory; if 1, distributed bag of words
    Postconditions:
    returns numpy array shape(2, vec_size) -- sentence embeddings using doc2vec
    
    Sources:
    https://cs.stanford.edu/~quocle/paragraph_vector.pdf
    https://rare-technologies.com/doc2vec-tutorial/
    https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
    https://radimrehurek.com/gensim/models/doc2vec.html
    """
    tagged_data = [TaggedDocument(words = word_tokenize(_d.lower()), tags = [str(i)]) for i, _d in enumerate(sentences)]
    model = Doc2Vec(vector_size = vec_size, alpha = alpha, min_alpha = 0.00025, min_count = 1, dm = model)
    model.build_vocab(tagged_data)
    
    for epoch in range(max_epochs):
        model.train(tagged_data, total_examples = model.corpus_count, epochs = epoch)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
        
    result = np.zeros(shape=(len(sentences), vec_size))
    for i, _d in enumerate(sentences):
        result[i] = model.docvecs[str(i)]
    
    return result

def convert_pretrained_doc2vec(sentences):
    pass
    
def convert_infersent():
    """https://github.com/facebookresearch/InferSent https://arxiv.org/abs/1705.02364"""
    pass

def convert_tfidf():
    pass

#TEST
#FIXME: REPLACE WITH PYTHON RECOMMENDED UNIT TESTS
if __name__ == "__main__":
    #Testing untrained doc2vec
    sentences = ["Hi my name is Mike.", "I am a programmer", "This is a really hard project", "Hopefully this works."]
    print(convert_doc2vec(sentences, max_epochs = 20, model = 0))
    print(convert_doc2vec(sentences, max_epochs = 20, model = 1))
    
    
    #Testing BOW
    # print(convert_bag_word_vector(['the', 'good', 'dog', 'is', 'red'], {"good", "dog", "red"}))
    # print(convert_bag_word_vector(['the', 'good', 'good', 'dog', 'is', 'red'], {"good", "dog", "red"}))
    # print(convert_bag_word_vector(['the', 'good', 'good', 'dog', 'is', 'red'], {"good", "dog", "red", "blue"}))
    # print(convert_bag_word_vector(['the', 'good', 'good', 'good', 'dog', 'is', 'red'], {"good", "dog", "red", "blue"}))
    print(convert_bow(sentences, {"hi", "name", "mike", "programmer", "hard", "project"}))  