from flask import Flask, request, jsonify
from botnoi import cv
import requests
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World !'

@app.route('/prediction')
def classifier():
    img_url = request.values['p_image_url']
    a = cv.image(img_url)
    feat = a.getmobilenet()
    model = joblib.load('model_mobilenet.pkl')
    
    #Predict classes with LinearSVC
    prediction = model.predict([feat])
    result = {'img_url': img_url, 'prediction': prediction[0]}
    return jsonify(result)


if __name__=='__main__':
    app.run(debug=True)



