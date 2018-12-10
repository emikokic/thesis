
import csv
import argparse
import ast
import numpy as np
import pandas as pd
import tensorflow as tf

import os
from os import path
from keras.models import Model
from keras.layers import Dense, Dropout, Input, TimeDistributed
from keras.layers import Conv1D, BatchNormalization, Embedding
from keras import regularizers, optimizers
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, CSVLogger
from gensim.models import KeyedVectors


NUM_CLASSES = 5  # PER - LOC - ORG - MISC - O
MAX_SEN_LEN = 30


def load_dataset():
    train_data = pd.read_csv('~/thesis/corpus_WiNER/seq_tag_instances/sen_entities_max_len_30_train.csv')
    val_data = pd.read_csv('~/thesis/corpus_WiNER/seq_tag_instances/sen_entities_max_len_30_val.csv')
    test_data = pd.read_csv('~/thesis/corpus_WiNER/seq_tag_instances/sen_entities_max_len_30_test.csv')

    return train_data, val_data, test_data


def transform_input(instances, mapping, maxlen):
    """Replaces the words in instances with their index in mapping.
    Args:
        instances: a list of text instances.
        mapping: an dictionary from words to indices.
    Returns:
        A matrix with shape (n_instances, m_words)."""
    word_indices = []
    for instance in instances:
        l = []
        for word in instance:
            try:
                l.append(mapping[word].index)
            except KeyError:
                l.append(0)  # index to '</s>' word vector
        word_indices.append(l)

    return pad_sequences(word_indices, maxlen=maxlen)


def tagToInt(tag):
    return {'PER': 0, 'LOC': 1, 'ORG': 2, 'MISC': 3, 'O': 4}[tag]


def preprocess_data(train_data, val_data, test_data, w2v_model, train_examples_rate, maxlen):

    train_examples_rate = int(train_data['sentence'].size * train_examples_rate)

    X_train = train_data['sentence'].map(lambda sen: np.array(ast.literal_eval(sen))).values[:train_examples_rate]
    y_train = train_data['entities'].map(lambda sen: np.array(ast.literal_eval(sen))).values[:train_examples_rate]
    X_val = val_data['sentence'].map(lambda sen: np.array(ast.literal_eval(sen))).values
    y_val = val_data['entities'].map(lambda sen: np.array(ast.literal_eval(sen))).values
    X_test = test_data['sentence'].map(lambda sen: np.array(ast.literal_eval(sen))).values
    y_test = test_data['entities'].map(lambda sen: np.array(ast.literal_eval(sen))).values

    def fun(y): return tagToInt(y)
    y_train = np.array([np.array(list(map(fun, labels))) for labels in y_train])
    y_val = np.array([np.array(list(map(fun, labels))) for labels in y_val])
    y_test = np.array([np.array(list(map(fun, labels))) for labels in y_test])

    X_train = transform_input(X_train, w2v_model.vocab, maxlen)
    X_val = transform_input(X_val, w2v_model.vocab, maxlen)
    X_test = transform_input(X_test, w2v_model.vocab, maxlen)

    # convert class vectors to binary class matrices
    y_train = np.array([to_categorical(labels, NUM_CLASSES) for labels in y_train])
    y_val = np.array([to_categorical(labels, NUM_CLASSES) for labels in y_val])
    y_test = np.array([to_categorical(labels, NUM_CLASSES) for labels in y_test])

    # padding target labels
    y_train = pad_sequences(y_train, maxlen=MAX_SEN_LEN,  # [0,0,0,0,1] one-hot encoding for 'O' label
                            value=np.array([0, 0, 0, 0, 1], dtype='float32'))
    y_val = pad_sequences(y_val, maxlen=MAX_SEN_LEN,  # [0,0,0,0,1] one-hot encoding for 'O' label
                          value=np.array([0, 0, 0, 0, 1], dtype='float32'))
    y_test = pad_sequences(y_test, maxlen=MAX_SEN_LEN,  # [0,0,0,0,1] one-hot encoding for 'O' label
                           value=np.array([0, 0, 0, 0, 1], dtype='float32'))

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(h_params, input_shape, w2v_model):

    inp = Input(shape=input_shape)
    emb = Embedding(len(w2v_model.vocab),        # Vocabulary size
                    w2v_model.vector_size,       # Embedding size
                    weights=[w2v_model.vectors],  # Word vectors
                    trainable=False  # This indicates the word vectors must not be changed
                                     # during training.
                    )(inp)
    conv1_1 = Conv1D(filters=h_params.conv_filters, kernel_size=2, padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(h_params.l2))(emb)

    for _ in range(h_params.conv_layers - 1):
        conv1_1 = Conv1D(filters=h_params.conv_filters, kernel_size=2, padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(h_params.l2))(conv1_1)

    drp1 = Dropout(h_params.drop)(conv1_1)
    out = TimeDistributed(Dense(NUM_CLASSES, activation="softmax"))(drp1)

    model = Model(inputs=inp, outputs=out)

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sequence tagging with supervised cnn depth model')
    parser.add_argument('--conv_layers',
                        default=1,
                        type=int,
                        help='Number of convolutional layers')
    parser.add_argument('--train_examples_rate',
                        default=1,
                        type=float,
                        help='1 -> 100%,  0.25 -> 25%, etc.')
    parser.add_argument('--conv_filters',
                        default=10,
                        type=int,
                        help='Number of convolution filters per kernel_size.')
    parser.add_argument('--drop',
                        default=0,
                        type=float,
                        help='Randomly fraction rate of input units to 0'
                             'at each update during training time')
    parser.add_argument('--l2',
                        default=0.1,
                        type=float,
                        help='L2 kernel regularizer')
    parser.add_argument('--batch_size',
                        default=512,
                        type=int,
                        help='Number of instances in each batch.')
    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='Number of training epochs.')

    args = parser.parse_args()

    train_examples_rate = args.train_examples_rate
    conv_filters = args.conv_filters
    drop = args.drop
    l2 = args.l2
    batch_size = args.batch_size
    epochs = args.epochs

    print('Loading dataset...')
    train_data, val_data, test_data = load_dataset()

    print('Loading word2vec model...')
    w2v_model = KeyedVectors.load('/home/ekokic/thesis/models/google/word2vecGoogle.model')

    print('Preprocessing input data...')
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(train_data, val_data,
                                                                     test_data, w2v_model,
                                                                     train_examples_rate,
                                                                     MAX_SEN_LEN)
    print('Building model...')
    input_shape = (X_train.shape[1], )
    model = build_model(args, input_shape, w2v_model)

    print('Compiling model...')
    model.compile(loss=categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=['accuracy'])

    print('Training model...')
    args = list(vars(args).items())
    experiment_name = 'seq_tag_cnn_depth_supervised'
    for key, value in args:
        experiment_name += ('_' + str(key) + '_' + str(value))
    if not os.path.exists('experiments' + '/' + experiment_name):
        os.makedirs('experiments' + '/' + experiment_name)

    callbacks = [
        # ModelCheckpoint(filepath=os.path.join('experiments', experiment_name,
        #                                       'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
        #                 monitor='val_acc',
        #                 verbose=1,
        #                 save_best_only=True,
        #                 save_weights_only=False,
        #                 mode='auto'),
        CSVLogger(filename=os.path.join('experiments', experiment_name, 'train_logs.csv'),
                  separator=',',
                  append=False)
    ]
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks)

    # print('Saving model predictions...')
    # predictions = model.predict(X_train)
    # np.savez_compressed(os.path.join('experiments', experiment_name, 'pred_labels_train'), predictions)
    # np.savez_compressed(os.path.join('experiments', experiment_name, 'true_labels_train'), y_train)
    # predictions = model.predict(X_val)
    # np.savez_compressed(os.path.join('experiments', experiment_name, 'pred_labels_val'), predictions)
    # np.savez_compressed(os.path.join('experiments', experiment_name, 'true_labels_val'), y_val)
    # predictions = model.predict(X_test)
    # np.savez_compressed(os.path.join('experiments', experiment_name, 'pred_labels_test'), predictions)
    # np.savez_compressed(os.path.join('experiments', experiment_name, 'true_labels_test'), y_test)
