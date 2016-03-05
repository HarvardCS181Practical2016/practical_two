# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train".
from itertools import chain

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import pandas as pd
import util
import operator

TRAIN_DIR = "train"
TEST_DIR = "test"

call_set = set([])

def add_to_set(tree):
    for el in tree.iter():
        call = el.tag
        call_set.add(call)


def create_data_matrix(start_index, end_index, dlls, good_calls, direc="train",):
    X = None
    classes = []
    ids = [] 
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break

        print 'Processing file [{0}]: {1}'.format(i, datafile)

        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]

        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))

        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        add_to_set(tree)
        this_row = call_feats(tree, dlls, good_calls)
        if X is None:
            X = this_row 
        else:
            X = np.vstack((X, this_row))

    return X, np.array(classes), ids

def call_feats(tree, dlls, good_calls):
    call_counter = {}
    dll_counter = {}
    for el in tree.iter():
        call = el.tag
        if call not in call_counter:
            call_counter[call] = 0
        else:
            call_counter[call] += 1

        if call == 'load_dll':
            dll_name = el.get('filename')
            if dll_name is not None:
                dll_name = dll_name.encode('utf-8').strip()
                if dll_name not in dll_counter:
                    dll_counter[dll_name] = 1
                else:
                    dll_counter[dll_name] += 1

    call_feat_array = np.zeros(len(good_calls))
    for i in range(len(good_calls)):
        call = good_calls[i]
        call_feat_array[i] = 0
        if call in call_counter:
            call_feat_array[i] = call_counter[call]

    dll_name_array = np.zeros(len(dlls))
    for i in range(len(dlls)):
        dll_name_array[i] = 1 if dlls[i] in dll_counter else 0

    return np.concatenate((call_feat_array, dll_name_array))

def create_list_of_dlls(start_index, end_index, direc="train"):
    dlls_loaded = set([])
    calls = set([])
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue
        if i >= end_index:
            break

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        for el in tree.iter():
            calls.add(el.tag)
            if el.tag == 'load_dll':
                dlls_name = el.get('filename')
                if dlls_name is not None:
                    dlls_loaded.add(dlls_name.encode('utf-8').strip())

    return list(dlls_loaded), list(calls)


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def categorization_accuracy(predicted_file, actual_file):
    predicted = pd.read_csv(predicted_file)
    actual = pd.read_csv(actual_file)

    n_predicted, col = predicted.values.shape
    n_actual, col = actual.values.shape

    if n_predicted != n_actual:
        print 'Predicted and actual data does not match', n_predicted, n_actual
        return

    class_predicted = predicted.Prediction.values
    class_actual = actual.clazz.values

    correct = n_actual - np.count_nonzero(class_predicted - class_actual)

    return float(correct) / float(n_actual)


## Feature extraction
def main():
    # get list of dlls loaded in all files
    dlls, calls = create_list_of_dlls(0, 3086, TRAIN_DIR)
    print 'Numbers of unique dlls loaded: ', len(dlls), len(calls)

    good_calls = calls

    # feature columns
    columns = good_calls + dlls

    # create training dataframe and save as train.csv
    X_train, t_train, train_ids = create_data_matrix(0, 3086, dlls, calls, TRAIN_DIR)

    train_df = pd.DataFrame(X_train, columns=columns)
    train_df['clazz'] = t_train
    train_df['Id'] = train_ids
    train_df.to_csv('train.csv', index=False)

    # create test dataframe and save as test.csv
    X_valid, t_valid, valid_ids = create_data_matrix(0, 3724, dlls, calls, TEST_DIR)

    train_df = pd.DataFrame(X_valid, columns=columns)
    train_df['Id'] = valid_ids
    train_df.to_csv('test.csv', index=False)

    print 'Data matrix (training set):'
    print X_train
    print 'Classes (training set):'
    print t_train


    # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).


def split_training_dataset(filename):
    df_train = pd.read_csv(filename)
    df1 = df_train[:1999]
    df2 = df_train[2000:3086]
    df3 = pd.DataFrame()

    df3['Id'] = df2['Id']
    df3['clazz'] = df2['clazz']

    df2 = df2.drop(['clazz'], axis=1)

    df1.to_csv('train_small.csv', index=False)
    df2.to_csv('test_small.csv', index=False)
    df3.to_csv('actual.csv', index=False)


if __name__ == "__main__":
    main()
    # categorization_accuracy('predicted_RF01.csv', 'actual_small.csv')
    # split_training_dataset('train.csv')

