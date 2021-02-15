from flask import Flask, request
import tensorflow as tf
import requests
import joblib
import numpy 

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World !'

@app.route('/prediction')
def classifier():
    img_url = request.values['img_url']
    print('recieve')
    img = tf.image.decode_jpeg(requests.get(img_url).content, channels=3, name="jpeg_reader")
    print('decode')
    resized_img = tf.image.resize(img, (224,224))
    print('resize')
    image = tf.expand_dims(resized_img, axis=0)
    print('expand')
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    features = model(image).numpy()
    print('extract')
    flat_features = features.flatten()

    model = joblib.load('model.pkl')
    prediction = model.predict([flat_features])
    result = {'img_url': img_url, 'prediction': prediction[0]}
    return result


if __name__=='__main__':
    app.run(debug=True)



