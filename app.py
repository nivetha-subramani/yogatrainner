from flask import Flask, request, render_template, flash

import pickle

import time

from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import random
import sys

import logging
import hashlib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecret'



# intialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CNN = None  # global variable to hold the CNN model

def load_model():
    global CNN
    if CNN is not None:
        return  # model already loaded
    script_dir = os.path.dirname(__file__)
    model_json_path = os.path.join(script_dir, 'models', 'CNN_structure.json')

    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    try:
        # load model
        CNN = tf.keras.models.model_from_json(model_json)

        # load and set model weights
        weights_path = os.path.join(script_dir, 'models', 'CNN_weights.pkl')
        with open(weights_path, 'rb') as weights_file:
            weights = pickle.load(weights_file)
            CNN.set_weights(weights)

        # compile model
        CNN.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def get_model_prediction(image_path):
    load_model()
    try:
        # load and preprocess the image
        img = Image.open(image_path).resize((224, 224))
        # convert grayscale images to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.expand_dims(np.array(img), axis=0)
        
        # predict using the CNN model
        prediction = CNN.predict(img_array)
        
        # interpret the prediction
        predicted_index = np.argmax(prediction[0])
        class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
        predicted_class = class_labels[predicted_index]
        return predicted_class
    except Exception as e:
        logger.error(f"Error in get_model_prediction: {e}")
        return None


scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('TL_model.pkl', 'rb'))

@app.route('/')
def homepage():
    return render_template('home.html')


@app.route('/diabetic', methods=['GET', 'POST'])
def home():
    prediction = -1
    if request.method == 'POST':
        pregs = int(request.form.get('pregs'))
        gluc = int(request.form.get('gluc'))
        bp = int(request.form.get('bp'))
        skin = int(request.form.get('skin'))
        insulin = float(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        func = float(request.form.get('func'))
        age = int(request.form.get('age'))

        input_features = [[pregs, gluc, bp, skin, insulin, bmi, func, age]]
        print(input_features)
        prediction = model.predict(scaler.transform(input_features))
        print(prediction)

        if prediction==1:
            time.sleep(4)
            return render_template('rec1.html')

        
    return render_template('index1.html', prediction=prediction)

@app.route('/indexmri')
def indexmri():
    return render_template('indexmri.html')

@app.route('/get-random-image', methods=['GET'])
def get_random_image():
    try:  
         # select a random directory and then a random image within the image directory
        class_dirs = ['glioma', 'meningioma', 'notumor', 'pituitary']
        selected_class = random.choice(class_dirs)
        image_dir = os.path.join('static', selected_class)
        image_name = random.choice(os.listdir(image_dir))
        image_path = os.path.join(image_dir, image_name)
        predicted_label = get_model_prediction(image_path)
        web_accessible_image_path = url_for('static', filename=f'{selected_class}/{image_name}')
        return jsonify({
            'image_path': web_accessible_image_path,
            'actual_label': selected_class,
            'predicted_label': predicted_label
        })
    except Exception as e:
        logger.error(f"Error in get-random-image route: {e}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)