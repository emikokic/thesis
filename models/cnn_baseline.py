








def load_dataset():
    train_data = pd.read_csv('./corpus_WiNER/cnn_instances/words_entity_W_2_cnn_train.csv')
    dev_data = pd.read_csv('./corpus_WiNER/cnn_instances/words_entity_W_2_cnn_dev.csv')
    test_data = pd.read_csv('./corpus_WiNER/cnn_instances/words_entity_W_2_cnn_test.csv')

    return train_data, dev_data, test_data


def transform_input(instances, mapping):
    """Replaces the words in instances with their index in mapping.
    Args:
        instances: a list of text instances.
        mapping: an dictionary from words to indices.
    Returns:
        A matrix with shape (n_instances, m_words)."""
    word_indices = []
    for instance in instances:
        l = []
        for word in ast.literal_eval(instance):
            try:
                l.append(mapping[word].index)
            except KeyError:
                l.append(0) # index to '</s>' word vector
        word_indices.append(l)
        
    return word_indices


def preprocess_data(train_data, dev_data, test_data, w2v_model):

    X_train = train_data['words'].values[:100000]
    y_train = train_data['entityType'].values[:100000]
    X_dev = dev_data['words'].values[:20000]
    y_dev = dev_data['entityType'].values[:20000]
    X_test = test_data['words'].values[:20000]
    y_test = test_data['entityType'].values[:20000]

    X_train = np.asarray(transform_input(X_train, w2v_model.vocab))
    X_dev = np.asarray(transform_input(X_dev, w2v_model.vocab))
    X_test = np.asarray(transform_input(X_test, w2v_model.vocab))

    NUM_CLASSES = 5 # PER - LOC - ORG - MISC - O
    y_train = [tagToInt(y) for y in y_train] # this transformation is needed to apply 
    y_dev = [tagToInt(y) for y in y_dev]     # to_categorical() keras method
    y_test = [tagToInt(y) for y in y_test]

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_dev = to_categorical(y_dev, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)


def build_model():

    # hyperparameters:
    conv_filters = 10
    pool_size = 2

    W = 2 # word window size

    input_shape = (2*W+1, ) # == X_train.shape[1]
'''shape: A shape tuple (integer), not including the batch size.
        For instance, `shape=(32,)` indicates that the expected input
        will be batches of 32-dimensional vectors.'''    
    inp = Input(shape=input_shape)

    emb = Embedding(len(w2v_model.vocab),  # Vocabulary size
                    w2v_model.vector_size, # Embedding size
                    weights=[w2v_model.vectors], # Word vectors
                    trainable=False  # This indicates the word vectors must not be changed
                                     # during training.
          )(inp)# The output here has shape (batch_size (?), words_in_reviews (?), embedding_size)

    # Specify each convolution layer and their kernel size i.e. n-grams 
    conv1_1 = Conv1D(filters=conv_filters, kernel_size=3, activation='relu')(emb)
    btch1_1 = BatchNormalization()(conv1_1)
    maxp1_1 = MaxPooling1D(pool_size=pool_size)(btch1_1)
    flat1_1 = Flatten()(maxp1_1)

    conv1_2 = Conv1D(filters=conv_filters, kernel_size=3, activation='relu')(emb)
    btch1_2 = BatchNormalization()(conv1_2)
    maxp1_2 = MaxPooling1D(pool_size=pool_size)(btch1_2)
    flat1_2 = Flatten()(maxp1_2)

    conv1_3 = Conv1D(filters=conv_filters, kernel_size=3, activation='relu')(emb)
    btch1_3 = BatchNormalization()(conv1_3)
    maxp1_3 = MaxPooling1D(pool_size=pool_size)(btch1_3)
    flat1_3 = Flatten()(maxp1_3)

    # Gather all convolution layers
    cnct = concatenate([flat1_1, flat1_2, flat1_3], axis=1)
    drp1 = Dropout(0)(cnct)

    dns1  = Dense(128, activation='relu')(drp1)
    out = Dense(num_classes, activation='softmax')(dns1)

def main():
    
    #TODO: definir argparser


    print('Loading dataset...')
    train_data, dev_data, test_data = load_dataset()

    print('Loading word2vec model...')
    w2v_model = KeyedVectors.load('./models/word2vecGoogle.model')
    
    print('Preprocessing input data...')
    X_train, X_dev, X_test, y_train, y_dev, y_test = preprocess_data(train_data, dev_data, 
                                                                     test_data, w2v_model)

    print('Building model...')


    print('Training...')


    print('Saving prediction...')


    print('Saving model...')













if __name__ == '__cnn_baseline__':
    main()

