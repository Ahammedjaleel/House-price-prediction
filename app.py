# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:32:25 2019

@author: Sabitha Jaleel
"""


import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

import lightgbm as lgb

app= Flask(__name__)
model = pickle.load(open('House_prediction.pkl','rb'))


@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods =['POST'])
def predict():
  int_features = [int(x) for x in request.form.values()]
  final_features = [np.array(int_features)]
  prediction = model.predict(final_features)

  output = round(prediction[0],2)
  return render_template('index.html',prediction_text = 'House price should be $  {}'.format(output))

if __name__ == '__main__':
  app.run(debug=True)
