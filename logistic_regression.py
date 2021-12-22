import numpy as np
import pandas as pd
import os
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
import math
from scipy.special import expit
import sys
from helper_functions import get_accuracy_precision_recall_f1_score, read_data_given_folder_and_label, get_bow_and_bernoulli



def train_logistic_regression(train_np_arr, label_arr, learning_rate, lambda_):
    weight_vector = np.zeros((1, train_np_arr.shape[1]))
    number_of_iterations = 100
    for i in range(number_of_iterations):
        prediction = expit(np.multiply(train_np_arr, weight_vector).sum(axis=1))
        diff_error = label_arr - prediction
        prediction_update = np.multiply(train_np_arr, diff_error.reshape(-1,1))
        weight_update = prediction_update.sum(axis = 0)
        new_weight_vector = weight_vector + ((learning_rate * weight_update) - (learning_rate * lambda_ * weight_vector))
        weight_vector = new_weight_vector.copy()

    return weight_vector

def test_logistic_regression(weights, test_data_array):
    prediction = expit(np.multiply(test_data_array[:, :-1], weights).sum(axis=1))
    pred = np.where(prediction>=0.5, 1,0)
    return pred



def run_logistic_regression(train_data, train_val_split, lambda_val, bernoulli_flag):
    split_param_index = int(train_data.shape[0] * 0.7)


    validation_data = train_data[split_param_index:].copy()
    train_data_ar = train_data[:split_param_index].copy()

    if bernoulli_flag==True:
        vectorizer = CountVectorizer(binary=True)
    else:
        vectorizer = CountVectorizer()

    bag_of_words = vectorizer.fit_transform(train_data_ar[:,0])
    bag_of_words = bag_of_words.toarray()

    labels = np.where(train_data_ar[:,1]=="ham", 1, 0).reshape(-1,1)
    bias_weight = np.ones((bag_of_words.shape[0],1))
    train_np_arr = np.hstack((bag_of_words, bias_weight))
    train_np_arr = np.hstack((train_np_arr, labels))

    accuracy = 0
    for lambda_ in lambda_val:
        weights = train_logistic_regression(train_np_arr[:,:-1], train_np_arr[:,-1], learning_rate = 0.01, lambda_ = lambda_)
        # vectorizing the test data
        validation_df = vectorizer.transform(validation_data[:,0]).toarray()
        # getting the labels
        labels = np.where(validation_data[:,1]=="ham", 1, 0).reshape(-1,1)
        # initializing the bias term to 1
        bias_weight = np.ones((validation_data.shape[0],1))

        # combining the test data with
        validation_df_arr = np.hstack((validation_df, bias_weight))
        validation_df_arr = np.hstack((validation_df_arr, labels))

        pred = test_logistic_regression(weights, validation_df_arr)
        current_accuracy = (pred == np.where(validation_data[:,1]=="ham", 1, 0)).sum()/pred.shape[0]
        # print(current_accuracy)
        if current_accuracy > accuracy:
            accuracy = current_accuracy
            weight_final = weights
            final_lamda = lambda_

    if bernoulli_flag==True:
        new_vectorizer = CountVectorizer(binary=True)
    else:
        new_vectorizer = CountVectorizer()

    bag_of_words_or_bernoulli = new_vectorizer.fit_transform(train_data[:,0]).toarray()
    bias_weight = np.ones((bag_of_words_or_bernoulli.shape[0],1))
    train_labels = np.where(train_data[:, 1]=="ham", 1, 0)
    train_np_arr = np.hstack((bag_of_words_or_bernoulli, bias_weight))
    final_weights  = train_logistic_regression(train_np_arr, train_labels, learning_rate = 0.01, lambda_ = final_lamda)
    return final_weights, new_vectorizer


if __name__ == '__main__':
    path = sys.argv[1]
    if len(sys.argv)==3:
        seed = int(sys.argv[2])
    else:
        seed = 1000
    train_data = read_data_given_folder_and_label(path, test = False)
    test_data = read_data_given_folder_and_label(path, test = True)
    np.random.seed(seed)
    np.random.shuffle(train_data)

    train_val_split = 0.7
    lambda_val = [0.0001, 0.001, 0.1, 1, 5]

    bernoulli_flag = False
    final_weights, new_vectorizer = run_logistic_regression(train_data, train_val_split, lambda_val, bernoulli_flag)

    # vectorizing the test data
    test_df = new_vectorizer.transform(test_data[:,0]).toarray()
    # getting the labels
    labels = np.where(test_data[:,1]=="ham", 1, 0).reshape(-1,1)
    # initializing the bias term to 1
    bias_weight = np.ones((test_df.shape[0],1))
    # combining the test data with
    test_data_np_arr = np.hstack((test_df, bias_weight))
    test_data_np_arr = np.hstack((test_data_np_arr, labels))
    pred = test_logistic_regression(final_weights, test_data_np_arr)
    # print((pred == np.where(test_data[:,1]=="ham", 1, 0)).sum()/pred.shape[0])
    test_labels = np.where(test_data[:,1]=="ham", 1, 0)
    accuracy,precision, recall, f1 = get_accuracy_precision_recall_f1_score(test_labels, pred)
    print("------------------------------------------------------------")
    print("Metrics of Logistic Regression on bag of words")
    print("Accuracy = ", accuracy)
    print("Precision = ", precision)
    print("Recall = ", recall)
    print("F1 Score = ", f1)



    bernoulli_flag = True
    final_weights, new_vectorizer = run_logistic_regression(train_data, train_val_split, lambda_val, bernoulli_flag)

    # vectorizing the test data
    test_df = new_vectorizer.transform(test_data[:,0]).toarray()
    # getting the labels
    labels = np.where(test_data[:,1]=="ham", 1, 0).reshape(-1,1)
    # initializing the bias term to 1
    bias_weight = np.ones((test_df.shape[0],1))
    # combining the test data with
    test_data_np_arr = np.hstack((test_df, bias_weight))
    test_data_np_arr = np.hstack((test_data_np_arr, labels))
    pred = test_logistic_regression(final_weights, test_data_np_arr)
    # print((pred == np.where(test_data[:,1]=="ham", 1, 0)).sum()/pred.shape[0])
    # test_labels = np.where(test_data[:,1]=="ham", 1, 0)
    accuracy,precision, recall, f1 = get_accuracy_precision_recall_f1_score(test_labels, pred)
    print("------------------------------------------------------------")
    print("Metrics of Logistic Regression on bernoulli data ")
    print("Accuracy = ", accuracy)
    print("Precision = ", precision)
    print("Recall = ", recall)
    print("F1 Score = ", f1)
