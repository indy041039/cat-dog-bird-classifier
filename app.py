from flask import Flask, request
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
    img_url = request.values['img_url']
    response = requests.get(img_url)
    image_bytes = io.BytesIO(response.content)
    img = PIL.Image.open(image_bytes)
    img = np.array(img)
    resized_img = tf.image.resize(img, (224,224))
    image = tf.expand_dims(resized_img, axis=0)
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    features = model(image).numpy()
    flat_features = features.flatten()

    model = joblib.load('model.pkl')
    prediction = model.predict([flat_features])
    result = {'img_url': img_url, 'prediction': prediction[0]}
    return result


if __name__=='__main__':
    app.run(debug=True)



