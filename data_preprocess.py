import datetime as dt
import os
import re
import sqlite3
from zipfile import ZipFile
import pandas as pd
from bs4 import BeautifulSoup
from kaggle.api.kaggle_api_extended import KaggleApi
from nltk import WordNetLemmatizer, stem
from nltk.corpus import stopwords as spw
from sqlalchemy import create_engine
from tqdm import tqdm


def download_imdb_reviews_kaggle():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('cryptexcode/mpst-movie-plot-synopses-with-tags')
    zf = ZipFile('mpst-movie-plot-synopses-with-tags.zip')
    zf.extractall()
    zf.close()
    data = pd.read_csv('mpst_full_data.csv')
    print("Data downloaded from Kaggle successfully")
    return data


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def mpst_database():
    start = dt.datetime.now()
    if not os.path.isfile('mpst.db'):
        print("Creating Database...")
        disk_engine = create_engine('sqlite:///mpst.db')
        start = dt.datetime.now()
        chunksize = 15000
        j = 0
        index_start = 1
        index_start = 1
        for df in pd.read_csv('mpst_full_data.csv', chunksize=chunksize, iterator=True, encoding='utf-8'):
            df.index += index_start
            j += 1
            df.to_sql('mpst_full_data', disk_engine, if_exists='append')
            index_start = df.index[-1] + 1
        print("Database created successfully")
    else:
        print("Database Already Exist.")
    print("Time taken to run this cell :", dt.datetime.now() - start)


'''Get SQLite 3 for querying
    database '''


def data_load():
    print("Connecting to mpst DB")
    con = sqlite3.connect('mpst.db')
    data_no_dup = pd.read_sql_query('SELECT title,plot_synopsis,tags,split,synopsis_source,COUNT(*) as cnt_dup FROM '
                                    'mpst_full_data GROUP BY title', con)
    con.close()

    data_no_dup["tag_count"] = data_no_dup["tags"].apply(lambda text: len(str(text).split(", ")))
    print("Data read from mpst db to dataframe")
    print(data_no_dup.head())
    return data_no_dup

stopwords = set(spw.words('english'))
sno = stem.SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def data_cleaning():
    # Data Cleaning
    print("Data cleaning process started")
    data_no_dup = data_load()
    preprocessed_synop = []
    for sentance in tqdm(data_no_dup['plot_synopsis'].values):
        sentance = re.sub(r"http\S+", "", sentance)
        sentance = BeautifulSoup(sentance, 'lxml').get_text()
        sentance = decontracted(sentance)
        sentance = re.sub("\S*\d\S*", "", sentance).strip()
        sentance = re.sub('[^A-Za-z]+', ' ', sentance)
        stemmed_sentence = []
        for e in sentance.split():
            if e.lower() not in stopwords:
                s = (sno.stem(lemmatizer.lemmatize(e.lower()))).encode('utf8')  # lemitizing and stemming each word
                stemmed_sentence.append(s)
        sentance = b' '.join(stemmed_sentence)
        preprocessed_synop.append(sentance)

    data_no_dup['CleanedSynopsis'] = preprocessed_synop  # adding a column of CleanedText which displays the data after
    # pre-processing of the review
    data_no_dup['CleanedSynopsis'] = data_no_dup['CleanedSynopsis'].str.decode("utf-8")

    data_no_dup.to_csv('data_with_all_tags.csv')
    print("Cleaned data successfully and csv written to directory")
    return data_no_dup


def preprocessed_synop(text_data):
    stopwords = set(spw.words('english'))
    sno = stem.SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    preprocessed_synop = []
    sentence = text_data
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    stemmed_sentence = []
    for e in sentence.split():
        if e.lower() not in stopwords:
            s = (sno.stem(lemmatizer.lemmatize(e.lower()))).encode('utf8')  # lemitizing and stemming each word
            stemmed_sentence.append(s)
    sentence = b' '.join(stemmed_sentence)
    preprocessed_synop.append(sentence)
    return preprocessed_synop


'''
Machine Learning Approach for Predicting Movie Tags
'''


def main():
    if os.path.exists("data_with_all_tags.csv"):
        print("Data already exists")
    else:
        print("Downloading dataset from Kaggle")
        data = download_imdb_reviews_kaggle()
        mpst_database()
        data_load()
        data_cleaning()


if __name__ == '__main__':
    main()
