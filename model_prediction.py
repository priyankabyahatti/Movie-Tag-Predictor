import nltk
import warnings
from nltk.stem import SnowballStemmer
import pickle
import sqlite3
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# from sklearn.metrics import f1_score,precision_score,recall_score,hamming_loss from keras.layers import Conv1D,
# Conv2D, Dense, Dropout, Flatten, LSTM, GlobalMaxPooling1D, MaxPooling2D, Activation, BatchNormalization
from sklearn.metrics import precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB


# nltk.download('punkt')
# nltk.download('wordnet')
# warnings.filterwarnings("ignore")
# stemmer = SnowballStemmer('english')


def getprocesseddata():
    data_with_all_tags = pd.read_csv("data_with_all_tags.csv")
    conn = sqlite3.connect('data.db')
    data_with_all_tags.to_sql('data', conn, if_exists='replace', index=False)
    train = pd.read_sql("Select * From data where split = 'train' OR split='val'", conn)
    test = pd.read_sql("Select * From data where split = 'test'", conn)
    conn.close()
    return train, test


def tokenize(x):
    x = x.split(',')
    tags = [i.strip() for i in x]  # Some tags contains whitespaces before them, so we need to strip them
    return tags


def predicttags(plotsynopsis):
    train, test = getprocesseddata()
    X_train = train["CleanedSynopsis"]
    y_train = train["tags"]
    X_test = test["CleanedSynopsis"]
    y_test = test["tags"]
    cnt_vectorizer = CountVectorizer(tokenizer=tokenize, max_features=6, binary='true').fit(y_train)
    y_train_multilabel = cnt_vectorizer.transform(y_train)
    y_test_multilabel = cnt_vectorizer.transform(y_test)
    # print(cnt_vectorizer.vocabulary_)
    # print(y_test_multilabel)
    # 1. TfidfVectorizer with 1 grams:
    tf_vectorizer = TfidfVectorizer(min_df=0.09, tokenizer=lambda x: x.split(" "), ngram_range=(1, 1))
    X_train_multilabel = tf_vectorizer.fit_transform(X_train)
    X_test_multilabel = tf_vectorizer.transform(X_test)
    # print("Dimensions of train data X:", X_train_multilabel.shape, "Y :", y_train_multilabel.shape)
    # print("Dimensions of test data X:", X_test_multilabel.shape, "Y:", y_test_multilabel.shape)
    # 1.1 OneVsRestClassifier + MultinomialNB:
    mb = MultinomialNB(class_prior=[0.5, 0.5])
    clf = OneVsRestClassifier(mb)
    clf.fit(X_train_multilabel, y_train_multilabel)
    xtest = [plotsynopsis]
    xtest_get = tf_vectorizer.transform(xtest)
    prediction1 = clf.predict(xtest_get)
    tags = cnt_vectorizer.inverse_transform(prediction1)[0]
    #   print("Predicted tag: ", cnt_vectorizer.inverse_transform(prediction1)[0])
    return tags


# prediction1 = clf.predict(X_test_multilabel)

# precision1 = precision_score(y_test_multilabel, prediction1, average='micro')
#
# recall1 = recall_score(y_test_multilabel, prediction1, average='micro')
#
# f1_score1 = 2 * ((precision1 * recall1) / (precision1 + recall1))

# print("precision1: {:.4f}, recall1: {:.4f}, F1-measure: {:.4f}".format(precision1, recall1, f1_score1))

# for i in range(5):
#     k = test.sample(1).index[0]
#     print("Movie: ", test['title'][k])
#     print("Actual genre: ", y_test[k])
#     print("Predicted tag: ", cnt_vectorizer.inverse_transform(prediction1[k])[0], "\n")

# k = 430
# print(k)
# print(prediction1[k])
# print("Movie: ", test['title'][k])
# print("Actual genre: ", y_test[k])
# print("Predicted tag: ", cnt_vectorizer.inverse_transform(prediction1[k])[0], "\n")

# print("Predicted tag: ", cnt_vectorizer.inverse_transform(prediction1)[0])
# print(*cnt_vectorizer.inverse_transform(prediction1)[0], sep='')
#
