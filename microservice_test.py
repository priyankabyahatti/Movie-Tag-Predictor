import psycopg2
import requests
import pytest
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to postgres sql database
def database_connection(database, user, password, host, port):
    conn = psycopg2.connect(database=database, user=user,
                            password=password, host=host, port=port)
    return conn.closed

#check if all the microservices are online
# Tag prediction microservice
"""
1. Preprocessor:3111
2. Database:3411:5432
3. Tag Predictor:3211
4. Webpage:3311
"""

def check_status(service ,url):
    status = False
    try:
       if service == "Database":
           if database_connection(os.getenv('DATABASE'), os.getenv('USER'), os.getenv('PASSWORD') ,os.getenv('HOST'), os.getenv('PORT')) == 0:
               status = True
       else:
        response = requests.get(url)
        if(response.status_code == 200):
            if(response.json()['status'] == 'success'):
                status = True
    except:
        status = False
    print(service, "is", "" if status else "Not", "Working" )
    return status


services = {"Preprocessor" : "http://localhost:3111/api/check-status",
"Database" : "http://localhost:3411",
"Tag Predictor" : "http://localhost:3211/api/check-status",
"Webpage" : "http://localhost:3311/api/check-status"}

def test_microservices_status():
    running = True
    for service, url in services.items():
        if not check_status(service, url):
            running = False
    assert running

if __name__ == '__main__':
    pytest.main()

