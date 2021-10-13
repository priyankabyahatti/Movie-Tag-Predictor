import pytest
import os
from data_preprocess import download_imdb_reviews_kaggle, decontracted


api_token = {"username":"nikhilkumaritaliya","key":"0af959f0af1b5b19d6671187d38ece5c"}

os.environ['KAGGLE_USERNAME'] = 'nikhilkumaritaliya'
os.environ['KAGGLE_KEY'] = '0af959f0af1b5b19d6671187d38ece5c'
# import json

# with open('/home/runner/.kaggle/kaggle.json', 'w') as file:
#     json.dump(api_token, file)

# !chmod 600 ~/.kaggle/kaggle.json

def test_data_download():
    data = download_imdb_reviews_kaggle()
    assert len(data) != 0, 'Data not downloaded from Kaggle'


# def test_create_db():
#     assert os.path.isfile('mpst.db'), 'Database not created'


# def test_data_load():
#     data = data_load()
#     assert len(data) != 0, 'Data not loaded from database'


def test_decontracted():
    phrase = "I won't do it"
    phrase = decontracted(phrase)
    assert phrase == 'I will not do it', 'decontracted function not working'


# def test_process_file_exist():
#     assert os.path.isfile('data_with_all_tags.csv'), 'Database not created'


if __name__ == '__main__':
    pytest.main()
