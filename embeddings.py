import numpy as np

def fill_w2vMatrix(sentence, sen_matrix, D, w2v_model):
    '''
    @return (sen_matrix, count) tuple where
    sen_matrix: filled matrix after apply w2v_model.
    no_vocab_words: words that are not in the vocabulary.
    '''
    no_vocab_words = []
    for idx, word in enumerate(sentence):
        try:
            vector = w2v_model.get_vector(word)
        except KeyError: # the word is not in the vocabulary
            vector = np.zeros((D)) # it's transformed to zeros vector
            no_vocab_words.append(word)
        sen_matrix[:,idx] = vector
    return sen_matrix, no_vocab_words


def concat_vectors(sentence, D, W, w2v_model):
    '''
    sentence: list of strings
    D: dimensionality of word vectors
    W: window
    w2v_model: word2vec model
    
    return vectors: L x 2WD matrix, where L is the length of the sentence.
    i.e. each row contains a word vector after apply concatenation strategy.
    '''
    L = len(sentence)
    shape = (D, L)
    sen_matrix = np.zeros(shape)
    vectors = np.zeros((L, 2*W*D))

    sen_matrix, no_vocab_words = fill_w2vMatrix(sentence, sen_matrix, D, w2v_model)
    
    # I: index of the target word
    for I in range(0, L):             
        concat_vec = []      
        # Padding with vector of zeros on the left
        if I - W < 0:
            concat_vec += list(np.zeros(abs(I-W) * D))
        # Concat vectors from the sentence
        for i in range(I - W, I + W + 1):             
            if i >= 0 and i != I and i < L:
                concat_vec += list(sen_matrix[:, i])          
        # Padding with vector of zeros on the right
        if I + W >= L:
            concat_vec += list(np.zeros(abs(I+W+1-L) * D))

        vectors[I,:] = np.asarray(concat_vec)
        
    return vectors, no_vocab_words


def mean_vectors(sentence, D, W, w2v_model):
    '''
    sentence: list of strings
    D: dimensionality of word vectors
    W: window
    w2v_model: word2vec model
    
    return vectors: L x D matrix, where L is the length of the sentence.
    i.e. each row contains a word vector after apply mean strategy.
    '''
    L = len(sentence)
    shape = (D, L)    
    sen_matrix = np.zeros(shape)
    vectors = np.zeros((L, D))
    
    sen_matrix, no_vocab_words = fill_w2vMatrix(sentence, sen_matrix, D, w2v_model)
    
    for I in range(L):
        vectors[I,:] = np.mean([sen_matrix[:,i] for i in range(I - W, I + W + 1) 
                                if i >= 0 and i != I and i < L], axis=0)       
    return vectors, no_vocab_words


def fractional_decay(sentence, D, W, w2v_model):
    '''
    sentence: list of strings
    D: dimensionality of word vectors
    W: window
    w2v_model: word2vec model
    
    return vectors: L x D matrix, where L is the length of the sentence.
    i.e. each row contains a word vector after apply fractional decay strategy.
    '''
    L = len(sentence)
    shape = (D, L)
    sen_matrix = np.zeros(shape)
    vectors = np.zeros((L, D))
    
    sen_matrix, no_vocab_words = fill_w2vMatrix(sentence, sen_matrix, D, w2v_model)
    
    for I in range(L):
        vectors[I,:] = np.sum([sen_matrix[:,j] * ((W - abs(I-j)) / W) 
                               for j in range(I - W, I + W + 1) 
                                if j >= 0 and j != I and j < L], axis=0)        
    return vectors, no_vocab_words


def exponential_decay(sentence, D, W, w2v_model):
    '''
    sentence: list of strings
    D: dimensionality of word vectors
    W: window
    w2v_model: word2vec model
    
    @return (vectors, no_vocab_words) tuple
    vectors: L x D matrix, where L is the length of the sentence.
    i.e. each row contains a word vector after apply exponential decay strategy.
    no_vocab_words: words that are not in the vocabulary.
    
    '''
    L = len(sentence)
    shape = (D, L)
    sen_matrix = np.zeros(shape)
    vectors = np.zeros((L, D))
    
    sen_matrix, no_vocab_words = fill_w2vMatrix(sentence, sen_matrix, D, w2v_model)    
    
    # Decay parameter alpha: We choose the parameter in such a way that the immediate words 
    # that surround the target word contribute 10 times more than the last words 
    # on both sides of the window.
    alpha = 1 - (0.1)**((W-1)**(-1))
    
    for I in range(L):
        vectors[I,:] = np.sum([sen_matrix[:,j] * ((1-alpha)**(abs(I-j)-1)) 
                               for j in range(I - W, I + W + 1) 
                                if j >= 0 and j != I and j < L], axis=0)       
        
    return vectors, no_vocab_words
