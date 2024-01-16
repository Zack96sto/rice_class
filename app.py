from flask import Flask, request, redirect, jsonify, render_template, url_for, session
import numpy as np
import cv2
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, firestore, db
from datetime import datetime
import random 
import string
from functools import wraps

app = Flask(__name__)
model = tf.keras.models.load_model('rice_image_classification.h5')
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Initialize Firebase
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://rice1234334-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

app.secret_key = 'your_secret_key'

users = {
    "admin": "password123"
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return "Invalid credentials"

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/ML')
@login_required
def ML():
    return render_template('ML.html')

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist("file")
    total_predictions = np.zeros(len(class_names))
    num_images = len(files)

    for file in files:
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(img, axis=0) / 255.
        predictions = model.predict(img_array)[0]
        total_predictions += predictions

    avg_predictions = total_predictions / num_images
    results = {class_name: str(prob) for class_name, prob in zip(class_names, avg_predictions)}
    return jsonify(results)


def generate_id(size=5, chars=string.ascii_letters + string.digits):
    """ Generate a random string of letters and digits """
    return ''.join(random.choice(chars) for _ in range(size))

@app.route('/upload', methods=['POST'])
def upload():
    data = request.json
    data["createdAt"] = datetime.now().isoformat()
    data["createdBy"] = "Admin"

    short_id = generate_id()  # Assuming you're still using the generate_id function

    ref = db.reference(f'rice_data/{short_id}')
    ref.set(data)

    return jsonify({"success": True, "id": short_id})


if __name__ == '__main__':
    app.run(debug=True)
