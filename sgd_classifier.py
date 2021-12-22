import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from helper_functions import get_accuracy_precision_recall_f1_score, read_data_given_folder_and_label, get_bow_and_bernoulli
from sklearn.exceptions import ConvergenceWarning

# sklearn generates warning Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
import warnings
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)


if __name__ == '__main__':
    path = sys.argv[1]
    if len(sys.argv)==3:
        seed = int(sys.argv[2])
    else:
        seed = np.random.randint(low = 0, high = 1000)
    train_data = read_data_given_folder_and_label(path, test = False)
    test_data = read_data_given_folder_and_label(path, test = True)
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    vectorizer = CountVectorizer()
    training_data = vectorizer.fit_transform(train_data[:,0])
    training_data = training_data.toarray()
    testing_data = vectorizer.transform(test_data[:,0]).toarray()
    train_label = np.where(train_data[:,1]=="ham", 1, 0)
    clf = GridSearchCV(estimator=SGDClassifier(), param_grid={'alpha': [0.001,0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'max_iter':[100,200,500,800,1000], 'loss':["hinge", "log"]},n_jobs=-1)
    clf.fit(training_data, train_label)
    print("Best Parameters are : ",clf.best_params_)
    pred_y = clf.predict(testing_data)
    true_y = np.where(test_data[:,1]=="ham", 1, 0)


    accuracy,precision, recall, f1 = get_accuracy_precision_recall_f1_score(true_y, pred_y)

    print("------------------------------------------------------------")
    print("Metrics of MultiNomial Logistic Regression ")
    print("Accuracy = ", accuracy)
    print("Precision = ", precision)
    print("Recall = ", recall)
    print("F1 Score = ", f1)
    print("\n")

    vectorizer = CountVectorizer(binary=True)
    training_data = vectorizer.fit_transform(train_data[:,0])
    training_data = training_data.toarray()
    testing_data = vectorizer.transform(test_data[:,0]).toarray()
    train_label = np.where(train_data[:,1]=="ham", 1, 0)
    clf = GridSearchCV(estimator=SGDClassifier(), param_grid={'alpha': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'max_iter':[100,200,500,800,1000], 'loss':["hinge", "log"]}, n_jobs=-1)

    clf.fit(training_data, train_label)
    print("Best Parameters are : ",clf.best_params_)
    pred_y = clf.predict(testing_data)
    true_y = np.where(test_data[:,1]=="ham", 1, 0)

    accuracy,precision, recall, f1 = get_accuracy_precision_recall_f1_score(true_y, pred_y)

    print("------------------------------------------------------------")
    print("Metrics of Bernoulli Logistic Regression ")
    print("Accuracy = ", accuracy)
    print("Precision = ", precision)
    print("Recall = ", recall)
    print("F1 Score = ", f1)
