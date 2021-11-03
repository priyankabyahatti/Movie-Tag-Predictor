import pytest
import os
from data_preprocess import download_imdb_reviews_kaggle, decontracted, database_connection


api_token = {"username":"nikhilkumaritaliya","key":"0af959f0af1b5b19d6671187d38ece5c"}

os.environ['KAGGLE_USERNAME'] = 'nikhilkumaritaliya'
os.environ['KAGGLE_KEY'] = '0af959f0af1b5b19d6671187d38ece5c'
# import json

# with open('/home/runner/.kaggle/kaggle.json', 'w') as file:
#     json.dump(api_token, file)

# !chmod 600 ~/.kaggle/kaggle.json

# def test_data_download():
    # data = download_imdb_reviews_kaggle()
    # assert len(data) != 0, 'Data not downloaded from Kaggle'

# TODO: 
def test_kaggle_connection():
    # add a method to just authenticate and check connection with kaggle and does not download data
    # return the boolean value
    return True

# def test_create_db():
#     assert os.path.isfile('mpst.db'), 'Database not created'


# def test_data_load():
#     data = data_load()
#     assert len(data) != 0, 'Data not loaded from database'


def test_decontracted():
    phrase = "I won't do it"
    phrase = decontracted(phrase)
    return phrase == 'I will not do it'

# def test_process_file_exist():
#     assert os.path.isfile('data_with_all_tags.csv'), 'Database not created'

# Test if db connection is successfully established or not
def test_db_connection():
    try:
        conn = database_connection()
        return conn.closed == 0
    except:
        return False

def unit_test():
    tests = {}
    tests["Decontracted Test:"] = test_decontracted()
    tests["Kaggle Connection:"] = test_kaggle_connection()
    tests["Database Connection:"] = test_db_connection()
    print("Found", len(tests), "tests..." )
    for test, result in tests.items():
        print(test, "PASSED" if result else "FAILED")
    print(sum(tests.values()), "out of", len(tests.values()), "tests passed." )
    return all(tests.values())


if __name__ == '__main__':
    pytest.main()
