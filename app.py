from flask import Flask, request
from botnoi import scrape as sc
from botnoi import cv
import joblib
import numpy 

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World !'

@app.route('/prediction')
def classifier():
    img_url = request.values['img_url']
    features = cv.image(img_url)
    features = features.getresnet50()
    model = joblib.load('img_cls.pkl')
    prediction = model.predict([features])
    result = {'img_url': img_url, 'prediction': prediction[0]}
    return result


if __name__=='__main__':
    app.run(debug=True)



