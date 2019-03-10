"""
Michael Altschuler
Created: 3/9/2019
Updated: 3/9/2019

main.py
driver program
"""

import sys
import preprocessing
import vector
import outline
import output

def get_input(filename):
    """get string from filename"""
    f = open(filename)
    result = f.read()
    f.close()
    return result

def preprocess_text(text):
    """preprocess text
    NOTE: all duplicate sentences need to be removed
    """
    sentences = preprocessing.tokenize_to_sentences(text)
    vocab = preprocessing.extract_vocab(text)
    return sentences, vocab

def vectorize(option, sentences, vocab):
    """convert sentences into vector form"""
    if option.lower() == "bow":
        return vector.convert_bow(sentences, vocab)
    elif option.lower() == "doc2vec":
        return vector.convert_doc2vec(sentences)
    else:
        raise "Not proper input for option"

def outline_vectors(sentences, vectors):
    """return Graph structure to heircharchically cluster the sentences"""
    return outline.kmeans_merge_clustering(sentences, vectors, 4)

def create_output(output_file, graph):
    """create html file"""
    output.create_output(output_file, graph)

def main(filename):
    """main/driver function"""
    # get input source
    text = get_input(filename)
    
    # preprocess text and extract vocab
    sentences, vocab = preprocess_text(text)
    
    # get 2d numpy array
    vectors = vectorize("bow", sentences, vocab)
    
    # get outline structure
    graph = outline_vectors(sentences, vectors)
    
    # print html file
    create_output("dummy3", graph)
    
    text = get_input(filename)
    sentences, vocab = preprocess_text(text)
    vectors = vectorize("doc2vec", sentences, vocab)
    graph = outline_vectors(sentences, vectors)
    create_output("dummy4", graph)

if __name__ == "__main__":
    main(sys.argv[1])