"""
Michael Altschuler
Created: 11/18/2018
Updated: 11/18/2018

vector.py
Convert preprocessed input into the proper vector representation

Pre-compilation:
pip install numpy
"""

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
#import skipthoughts

def convert_bag_word_vector(sentence, vocab):
    """convert each sentence to a bag of word vector
    where each component is a word from the vocab and the element is the count of the word
    Preconditions: a list of words from the sentence, a set of all possible vocab words
    Postconditions: a numpy array"""
    result = np.zeros(len(vocab), dtype = int)
    
    sorted_vocab = sorted(vocab)
    
    for x in range(len(sorted_vocab)):
        for s in sentence:
            if sorted_vocab[x] == s:
                result[x] = result[x] + 1
    return result

def convert_bow(sentences, vocab):
    result = np.zeros(dtype = int, shape=(len(sentences), len(vocab)))
    
    for i, s in enumerate(sentences):
        print(word_tokenize(s))
        result[i] = convert_bag_word_vector(word_tokenize(s.lower()), vocab)
    
    return result

def convert_skip_thought_embeddings():
    """convert to skip thoughts based off https://github.com/ryankiros/skip-thoughts"""
    #FIXME: COULD NOT SYNC UP WITH PYTHON3 REACHED critical error
    pass

def convert_doc2vec(sentences, max_epochs = 100, vec_size = 100, alpha = 0.05, model = 0):
    """convert to doc2vec for sentences https://cs.stanford.edu/~quocle/paragraph_vector.pdf and https://rare-technologies.com/doc2vec-tutorial/"""
    """https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5"""
    """Preconditions: sentences --> a list of sentences from the document
    Postconditions:"""
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
    # is pre training a model on a larger dataset necessary
    pass
    
def convert_infersent():
    """convert to InferSent https://github.com/facebookresearch/InferSent https://arxiv.org/abs/1705.02364"""
    pass

def convert_tfidf
    pass

#TEST
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