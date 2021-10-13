import os
import re
from sys import path
from zipfile import ZipFile
import pandas as pd
from bs4 import BeautifulSoup
from kaggle.api.kaggle_api_extended import KaggleApi
from nltk import WordNetLemmatizer, stem
from nltk.corpus import stopwords as spw
from tqdm import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import psycopg2

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
PASSWORD = 'MOVIES'

def API(Conf):
    print('In API selction')
    app.run(host='0.0.0.0', port=3111)


#database connection
def database_connection():
    conn = psycopg2.connect(database="tags_database", user="admin",
                            password="admin", host="tags_database", port="3411")
    return conn


def download_imdb_reviews_kaggle():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('cryptexcode/mpst-movie-plot-synopses-with-tags')
    zf = ZipFile('mpst-movie-plot-synopses-with-tags.zip')
    zf.extractall()
    zf.close()
    data = pd.read_csv('mpst_full_data.csv')
    print("Data downloaded from Kaggle successfully")
    return data.head(15000)


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


# def mpst_database():
#     start = dt.datetime.now()
#     if not os.path.isfile('mpst.db'):
#         print("Creating Database...")
#         disk_engine = create_engine('sqlite:///mpst.db')
#         start = dt.datetime.now()
#         chunksize = 15000
#         j = 0
#         index_start = 1
#         index_start = 1
#         for df in pd.read_csv('mpst_full_data.csv', chunksize=chunksize, iterator=True, encoding='utf-8'):
#             df.index += index_start
#             j += 1
#             df.to_sql('mpst_full_data', disk_engine, if_exists='append')
#             index_start = df.index[-1] + 1
#         print("Database created successfully")
#     else:
#         print("Database Already Exist.")
#     print("Time taken to run this cell :", dt.datetime.now() - start)


'''Get SQLite 3 for querying
    database '''

def creat_database():
    print("Creating Database...")
    conn = database_connection()
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS tags_table(
        idx integer,
        title text,
        plot_synopsis text,
        tags text,
        split text,
        synopsis_source text,
        cnt_dup integer,
        tag_count integer,
        CleanedSynopsis text);""")
    conn.commit()
    cursor.execute(
        """SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'tags_table'""")
    is_table = cursor.fetchall()
    print("tags_table created successfully in PostgreSQL ")
    conn.close()
    return is_table

# def data_load():
#     print("Connecting to mpst DB")
#     con = sqlite3.connect('mpst.db')
#     data_no_dup = pd.read_sql_query('SELECT title,plot_synopsis,tags,split,synopsis_source,COUNT(*) as cnt_dup FROM '
#                                     'mpst_full_data GROUP BY title', con)
#     con.close()

#     data_no_dup["tag_count"] = data_no_dup["tags"].apply(lambda text: len(str(text).split(", ")))
#     print("Data read from mpst db to dataframe")
#     print(data_no_dup.head())
#     return data_no_dup

stopwords = set(spw.words('english'))
sno = stem.SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()


def load_data(cleaned_data):
    print("Loading data to database")
    conn = database_connection()
    cur = conn.cursor()
    cur.execute("""DELETE FROM tags_table""")
    print(cleaned_data.columns)
    for row in cleaned_data.itertuples():
        cur.execute("""INSERT INTO tags_table VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (row[0], row.title, row.plot_synopsis, row.tags, row.split, row.synopsis_source, row.cnt_dup, row.tag_count, row.CleanedSynopsis))
    conn.commit()
    print("table loaded successfully")
    

def data_cleaning(data_no_dup):
    # Data Cleaning
    print("Data cleaning process started")
    data_no_dup['cnt_dup'] = data_no_dup['title'].groupby(data_no_dup['title']).transform('count')
    # data_no_dup = pd.read_csv('mpst_full_data.csv')
    data_no_dup["tag_count"] = data_no_dup["tags"].apply(lambda text: len(str(text).split(", ")))
    stopwords = set(spw.words('english'))
    sno = stem.SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

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
    # loading data to the database
    load_data(data_no_dup)
    return stopwords, sno, lemmatizer


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

    return ''.join(str(preprocessed_synop))


'''
Machine Learning Approach for Predicting Movie Tags
'''

# function to clean data
@app.route("/api/clean-data/<password>", methods=['GET'])
def clean_data(password):
    if password == PASSWORD:
        print("Data cleaning process started")
        main()
        # func = request.environ.get('werkzeug.server.shutdown')
        # if func is None:
        #     raise RuntimeError('Not running with the Werkzeug Server')
        # func()
        return "Data cleaning completed successfully"
    else:
        print("Wrong password")
        return "Invalid Password"


# function to preprocess data
@app.route("/api/preprocess-data/<password>", methods=['GET', 'POST'])
def preprocess_data(password):
    if password == PASSWORD:
        content = request.json
        data = content['data']
        print("recieved data")
        print(data)
        synop = preprocessed_synop(data) 
        return jsonify({"data": synop})
    else:
        return jsonify({"data": "Invalid Password"})


def main():
    if os.path.exists("data_with_all_tags.csv"):
        print("Data already exists")
    else:
        print("Downloading dataset from Kaggle")
        creat_database()
        data = download_imdb_reviews_kaggle()
        # mpst_database()
        # data_load()
        data_cleaning(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3111, debug=True)
    # config = {"preprocessor": "at 3111"}
    # p = multiprocessing.Process(target=API, args=(config))
    # p.start()
    # print("Preprocess started")
    # main()
    # p.terminate()
    # p.join()
    # print("Preprocess stopped")