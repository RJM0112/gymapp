
import flask
from flask import Flask, request , jsonify, render_template
#import jinja2
import numpy as np
import pandas as pd
import pickle

from flask import Flask, request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    arr = np.array([[data1, data2, data3]])
    prediction = model.predict(arr)
    return render_template('after.html',data=prediction)


if __name__ == "__main__":
    app.run(debug=True)
