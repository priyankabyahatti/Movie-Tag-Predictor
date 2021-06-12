from flask import Flask, render_template, request
from data_preprocess import preprocessed_synop
from model_prediction import predicttags

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('movie-tags-page.html')


@app.route('/', methods=['POST'])
def submit():
    my_data = request.form['plotinput']
    print(my_data)
    cleaned_synopsis = preprocessed_synop(my_data)
    print(cleaned_synopsis)
    plotdata = ''.join(str(cleaned_synopsis))
    tags = predicttags(plotdata)
    tags = '#' + ' #'.join(tags)
    return render_template('movie-tags-page.html', tags=tags)

    # 'You entered: {}'.format(tags)


if __name__ == "__main__":
    app.run(debug=True)
