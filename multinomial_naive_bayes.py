import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from helper_functions import get_accuracy_precision_recall_f1_score, read_data_given_folder_and_label, get_bow_and_bernoulli


def train_MultiNomialNB(train_data, corpus_unique_words_df):
    classes = ['ham','spam']
    priors = pd.DataFrame(train_data[:, 1]).value_counts(normalize = True)
    priors = priors.to_numpy()
    spam_ham_con_prob = []
    for each_class in classes:
        spam_ham_text = train_data[np.where(train_data[:,1]==each_class)]
        spam_ham_vectorizer = CountVectorizer()
        spam_ham_count_vector = spam_ham_vectorizer.fit_transform(spam_ham_text[:,0]).toarray()

        sum_of_spam_ham_words_col_wise = spam_ham_count_vector.sum(axis=0).reshape(-1,1)
        spam_ham_unique_words = np.array(spam_ham_vectorizer.get_feature_names()).reshape(-1,1).astype('object')
        T_ct = np.hstack((spam_ham_unique_words, sum_of_spam_ham_words_col_wise))


        spam_ham_count_df = pd.DataFrame(T_ct)
        T_ct_with_v_for_spam_ham = corpus_unique_words_df.merge(spam_ham_count_df,how='left').fillna(0)
        T_ct_with_v_for_spam_ham.columns = ['words', 'frequency']
        st = 'conditional_prob_given_'+ each_class
        T_ct_with_v_for_spam_ham[st] = (T_ct_with_v_for_spam_ham['frequency'] + 1) / (T_ct_with_v_for_spam_ham['frequency'].sum() + corpus_unique_words_df.shape[0])
        T_ct_with_v_for_spam_ham.drop('frequency', axis=1, inplace = True)
        spam_ham_con_prob.append(T_ct_with_v_for_spam_ham)



    conditional_prob_mat = spam_ham_con_prob[0].merge(spam_ham_con_prob[1])
    conditional_prob_mat['conditional_prob_given_ham'] = np.log(conditional_prob_mat['conditional_prob_given_ham'])
    conditional_prob_mat['conditional_prob_given_spam'] = np.log(conditional_prob_mat['conditional_prob_given_spam'])

    return conditional_prob_mat, priors



def test_MultiNomialNB(conditional_prob_mat, test_data, priors):
    predict = []
    true = []
    for data in test_data:
        words = data[0].split(' ')
        true.append(1 if data[1]=="ham" else 0)

        common_words_conditional_prob_table = conditional_prob_mat[conditional_prob_mat['words'].isin(words)].reset_index(drop=True)

        class_and_conditional_prob = common_words_conditional_prob_table.sum(axis = 0)[1:]
        class_and_conditional_prob[0] += np.log(priors[0])
        class_and_conditional_prob[1] += np.log(priors[1])
        if (class_and_conditional_prob[0]>class_and_conditional_prob[1]):
            predict.append(1)
        else:
            predict.append(0)

    prediction_df = pd.DataFrame([predict,true]).T
    return prediction_df



if __name__ == '__main__':
    path = sys.argv[1]
    train_data = read_data_given_folder_and_label(path, test = False)
    test_data = read_data_given_folder_and_label(path, test = True)

    bag_of_words, corpus_unique_words_df = get_bow_and_bernoulli(train_data, False)


    conditional_prob_mat_, priors = train_MultiNomialNB(train_data, corpus_unique_words_df)
    pred_df = test_MultiNomialNB(conditional_prob_mat_, test_data, priors)

    accuracy,precision, recall, f1 = get_accuracy_precision_recall_f1_score(pred_df[0], pred_df[1])
    print("------------------------------------------------------------")
    print("Metrics of Multinomial Naive Bayes ")
    print("Accuracy = ", accuracy)
    print("Precision = ", precision)
    print("Recall = ", recall)
    print("F1 Score = ", f1)
