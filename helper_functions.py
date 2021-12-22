import numpy as np
import os
import string
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def read_data_given_folder_and_label(folder_path, test):
    if test == True:
        path_sub = 'test/'
    else:
        path_sub = 'train/'

    label = ['ham', 'spam']
    return_data = []
    for ham_spam in label:
        data = []
        labels_arr = []
        folder_path_new = folder_path + path_sub + ham_spam + '/'
        print("Reading from ", folder_path_new)
        for filename in os.listdir(folder_path_new):
            string_read = open(folder_path_new+ filename, 'r', errors='ignore').read()
            string_read = string_read.translate(str.maketrans('', '', string.punctuation))
            string_read = re.sub("[^a-zA-Z'-]+", ' ', string_read)
            data.append(string_read)
            labels_arr.append(ham_spam)
        return_data.append(np.column_stack((data, labels_arr)))
        print("Files loaded")

    final_data = np.vstack((return_data[0],return_data[1]))

    return final_data


def get_bow_and_bernoulli(train_data, bernoulli_flag):
    if bernoulli_flag == True:
        vectorizer = CountVectorizer(binary=True)
    else:
        vectorizer = CountVectorizer()

    np_arr = vectorizer.fit_transform(train_data[:,0])
    bag_of_words = np_arr.toarray()
    corpus_unique_words_df = pd.DataFrame(vectorizer.get_feature_names())
    return bag_of_words, corpus_unique_words_df

def get_accuracy_precision_recall_f1_score(true_y, pred_y):
    accuracy = accuracy_score(true_y, pred_y)
    precision = precision_score(true_y, pred_y)
    recall = recall_score(true_y, pred_y)
    f1 = f1_score(true_y, pred_y)
    return accuracy, precision, recall, f1
