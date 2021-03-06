
from __future__ import absolute_import, print_function

import numpy as np
import pandas as pd
import ast
from keras.utils import to_categorical
from gensim.models import KeyedVectors


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
                l.append(0)  # index to '</s>' word vector
        word_indices.append(l)

    return word_indices


def tagToInt(tag):
    return {'PER': 0, 'LOC': 1, 'ORG': 2, 'MISC': 3, 'O': 4}[tag]


def preprocess_data(train_data, val_data, test_data, w2v_model, n_classes):

    X_train = train_data['words'].values
    y_train = train_data['entityType'].values
    X_val = val_data['words'].values[:20000]
    y_val = val_data['entityType'].values[:20000]
    X_test = test_data['words'].values[:20000]
    y_test = test_data['entityType'].values[:20000]

    X_train = np.asarray(transform_input(X_train, w2v_model.vocab))
    X_val = np.asarray(transform_input(X_val, w2v_model.vocab))
    X_test = np.asarray(transform_input(X_test, w2v_model.vocab))

    y_train = [tagToInt(y) for y in y_train]  # this transformation is needed to apply
    y_val = [tagToInt(y) for y in y_val]     # to_categorical() keras method
    y_test = [tagToInt(y) for y in y_test]

    # convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, n_classes)
    y_val = to_categorical(y_val, n_classes)
    y_test = to_categorical(y_test, n_classes)

    return X_train, X_val, X_test, y_train, y_val, y_test


class DataSet(object):

    def __init__(self, instances, labels, fake_data=False):
        if fake_data:
            self._num_examples = 1000
        else:
            assert len(labels) == 0 or instances.shape[0] == labels.shape[0], (
                "instances.shape: %s labels.shape: %s" % (instances.shape,
                                                          labels.shape))
            self._num_examples = instances.shape[0]

        self._instances = instances
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def instances(self):
        return self._instances

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_instance = [0 for _ in range(self._MAX_FAKE_SENTENCE_LEN)]
            fake_label = 0
            return ([fake_instance for _ in range(batch_size)],
                    [fake_label for _ in range(batch_size)])
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._instances = self._instances[perm]
            if len(self._labels) > 0:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        if len(self._labels) > 0:
            return self._instances[start:end], self._labels[start:end]
        else:
            return self._instances[start:end], None


class SemiDataSet(object):
    def __init__(self, instances, labels, n_labeled, n_classes):
        self.n_labeled = n_labeled

        # # Unlabled DataSet
        # self.unlabeled_ds = DataSet(instances, [])
        # self.num_examples = self.unlabeled_ds.num_examples

        # # Labeled DataSet
        # indices = np.arange(self.num_examples)
        # shuffled_indices = np.random.permutation(indices)
        # instances = instances[shuffled_indices]
        # labels = labels[shuffled_indices]
        # y = np.array([np.arange(n_classes)[lbl == 1][0] for lbl in labels])
        # n_from_each_class = n_labeled // n_classes
        # i_labeled = []
        # for c in range(n_classes):
        #     i = indices[y == c][:n_from_each_class]
        #     i_labeled += list(i)
        # l_instances = instances[i_labeled]
        # l_labels = labels[i_labeled]
        # self.labeled_ds = DataSet(l_instances, l_labels)

        # Experimento 3 ##### (descomentar lo siguiente solo si se quiere excluir que
        # los datos anotados se utilicen como anotados)
        # Unlabled DataSet
        self.unlabeled_ds = DataSet(instances[n_labeled:], [])
        self.num_examples = instances.shape[0]  # self.unlabeled_ds.num_examples

        # Labeled DataSet
        l_instances = instances[0:n_labeled]
        l_labels = labels[0:n_labeled]
        self.labeled_ds = DataSet(l_instances, l_labels)

    def next_batch(self, batch_size):
        unlabeled_instances, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_instances, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_instances, labels = self.labeled_ds.next_batch(batch_size)
        return labeled_instances, labels, unlabeled_instances


def read_data_sets(data_path, n_classes, n_labeled=100, fake_data=False, maxlen=None):
    class DataSets(object):
        pass

    data_sets = DataSets()

    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets

    print('Loading dataset...')
    train_data = pd.read_csv('%swords_entity_W_2_cnn_train_exp_3.csv' % data_path)
    val_data = pd.read_csv('%swords_entity_W_2_cnn_dev.csv' % data_path)
    test_data = pd.read_csv('%swords_entity_W_2_cnn_test.csv' % data_path)

    print('Loading word2vec model...')
    w2v_model = KeyedVectors.load('/home/ekokic/thesis/models/google/word2vecGoogle.model')

    print('Preprocessing input data...')
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(train_data, val_data,
                                                                     test_data, w2v_model, n_classes)

    data_sets.train = SemiDataSet(X_train, y_train, n_labeled, n_classes)
    data_sets.validation = DataSet(X_val, y_val)
    data_sets.test = DataSet(X_test, y_test)

    return data_sets, w2v_model
