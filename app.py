from flask import Flask, request, jsonify
from botnoi import cv
import tensorflow as tf
import requests
import joblib
import numpy as np
import io
import PIL

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World !'

@app.route('/prediction')
def classifier():
    img_url = request.values['p_image_url']
    #Convert image to numpy array
    # response = requests.get(img_url)
    # image_bytes = io.BytesIO(response.content)
    # img = PIL.Image.open(image_bytes)
    # img = np.array(img)
    # #Resize image into 224 x 224 x 3
    # resized_img = tf.image.resize(img, (224,224))
    # image = tf.expand_dims(resized_img, axis=0)
    # model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    # #Extract features with ResNet50
    # features = model(image).numpy()
    # flat_features = features.flatten()
    a = cv.image(img_url)
    feat = a.getmobilenet()
    model = joblib.load('model_mobilenet.pkl')
    #Predict classes with LinearSVC
    prediction = model.predict([feat])
    print(feat.shape)
    result = {'img_url': img_url, 'prediction': prediction[0]}
    return jsonify(result)


if __name__=='__main__':
    app.run(debug=True)



