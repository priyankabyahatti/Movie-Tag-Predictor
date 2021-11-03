from flask import Flask, render_template, request, jsonify
import flask
from nltk.util import pr
from flask_cors import CORS, cross_origin
# from data_preprocess import preprocessed_synop
import requests
import psycopg2
import time
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


#database connection
def database_connection():
    conn = psycopg2.connect(database="tags_database", user="admin",
                            password="admin", host="tags_database", port="5432")
    return conn


@app.route('/')
@cross_origin()
def hello_world():
    return render_template('movie-tags-page.html')

# check if application is running
@app.route("/api/check-status", methods=['GET'])
@cross_origin()
def check_status():
    return jsonify({"status": "success"})


@app.route('/', methods=['POST'])
@cross_origin()
def submit():
    my_data = request.form['plotinput']
    tags = get_synopsis(my_data)
    # tags = predicttags(plotdata)
    return render_template('movie-tags-page.html', tags=tags)

    # 'You entered: {}'.format(tags)
@app.route('/get-synopsis', methods=['POST', 'OPTIONS'])
@cross_origin()
def synopsis():
    my_data = request.data
    tags = get_synopsis(my_data.decode('utf-8'))
    return jsonify({"res": "success", "tags":tags})


def get_synopsis(my_data):
    # run api request to get cleaned synops
    clean_res = requests.post('http://preprocessor:3111/api/preprocess-data/MOVIES', json={"data":my_data})
    # clean_res.headers.add('Access-Control-Allow-Origin', '*')
    plotdata = ''
    if clean_res.ok:
        plotdata = clean_res.json()['data']
    # plotdata = ''.join(str(cleaned_synopsis))

    # run api request to get predicted tags
    tags_res = requests.post('http://prediction:3211/api/predict-tags/tags', json={"data":plotdata})
    tags = ''
    if tags_res.ok:
        tags = tags_res.json()['data']
    
    return tags

def check_status(service ,url):
    status = False
    try:
       if service == "Database:":
           if database_connection().closed == 0:
               status = True
       else:
            response = requests.get(url)
            if(response.status_code == 200):
                if(response.json()['status'] == 'success'):
                    status = True
    except:
        status = False
    print(service, "PASSED" if status else "FAILED")
    return status

def integration_failed():
    print("Error occured while testing.\n---------------------------- INTEGRATION TESTING ENDED -------------------------- \nStopping the Service.")


def integration_passed():
    print("Integration Test Passed.\n ---------------------------- INTEGRATION TESTING ENDED -------------------------- \nStarting the Service.")



if __name__ == "__main__":
    # check all the required services
    print("---------------------------- INTEGRATION TESTING STARTED --------------------------")
    has_failed = False
    services = {"Preprocessor:" : "http://preprocessor:3111/api/check-status",
    "Database:" : "http://tags_database:3411",
    "Tag Predictor:" : "http://prediction:3211/api/check-status"}
    res_services = {}
    for service, url in services.items():
        res_services[service] = check_status(service, url)

    if not all(res_services.values()):
        has_failed = True
        integration_failed()
    else:
            # load the data
        is_table = 0
        try:
            print("Checking data in database")
            conn = database_connection()
            cursor = conn.cursor()
            cursor.execute("select count(*) from tags_table")
            is_table = cursor.fetchall()[0][0]
            conn.close()
        except:
            print("Data loading failed")
        if is_table < 5:
            print("No data found! \nLoading the data... \nThis will take around 5 minutes...")
            try:
                r = requests.get('http://preprocessor:3111/api/clean-data/MOVIES', verify=False)
                print("Data loaded successfully")
            except:
                integration_failed()
                has_failed = True

    # run a api call with dummy data
    if not has_failed:
        with open("test_synopsis.txt", "r") as f:
            test_synop = f.read()
        try:
            tags = get_synopsis(test_synop)
            if not tags.strip():
                integration_failed()
                has_failed = True
            else:
                integration_passed()
                has_failed = False

                # open port
                app.run(host='0.0.0.0', port=3311,debug=True, threaded=True)
        except:
            integration_failed()
            has_failed = True

        


