import numpy as np
import pandas as pd
import pickle
import sys
import random, os

from PIL import Image, ExifTags
# Flask utils
from flask import Flask, redirect, render_template, request, url_for, session
from gevent.pywsgi import WSGIServer
from keras.applications.xception import preprocess_input
# Keras
# from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

# Define a flask app
app = Flask(__name__)

app.secret_key = 'galaxy'

def choose_file(df):
    single_im = df.sample(n=1)
    filename = single_im.Filename
    filename = filename.iloc[0]

    return filename

def img_predict(img, model):
    img = load_img(img, target_size=(299, 299))

    # Preprocessing the image
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    session.clear()
    return render_template('home.html')

@app.route('/about', methods=['GET'])
def about():
    # about project with link to github
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact_info():
    # contact info page
    return render_template('contact.html')

@app.route('/test', methods=['GET','POST'])
def testing():
    filename = choose_file(df)
    session['f'] = filename

    return render_template('test.html',filename=filename)

@app.route('/predict', methods=['GET', 'POST'])
def end_result():
    if request.method == 'POST':
        # Get the post request
        requested = request.form['class']
        filename = session.get('f', None)

        # Make prediction
        preds = img_predict('static/' + filename, model)

        # Process your result
        comp_guess = preds.argsort()[0][::-1][:1] # return top prediction
        name = class_names[comp_guess]
        percent = preds[0][[comp_guess]]*100
        top_pred = '<br>'.join([f'{name} - {percent:.2f}% certainty' for name, percent in zip(name,percent)])

        #get actual results
        actual = filename.split('/')[0]

        # result
        if (requested == actual) and (name[0] == actual):
            result = 'Tie: you and the computer were both right'
            score[0]+=1
            score[1]+=1
        elif (requested == actual) and (name[0] != actual):
            result = 'You win!'
            score[0]+=1
        elif (requested != actual) and (name[0] == actual):
            result = 'You missed this one, try again!'
            score[1]+=1
        else:
            result = 'You and the computer missed this one, try again!'


    return render_template('predict.html', requested=requested,top_pred=top_pred,filename=filename, actual=actual, result=result, current_score=current_score)

if __name__ == '__main__':
    MODEL_PATH = 'models/checkpoint.hdf5'
    with open ('models/class_names.pkl', 'rb') as f:
        class_names = np.array(pickle.load(f))

    df = pd.read_csv('models/results.csv')

    model = load_model(MODEL_PATH)
    model._make_predict_function()
    print('Model loaded. Start serving...')

    # app.run(port=8080, debug=True, host = "0.0.0.0")

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0',8080), app)
    http_server.serve_forever()
