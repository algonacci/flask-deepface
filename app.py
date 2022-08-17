import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2
from deepface import DeepFace
from PIL import Image

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

haar = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            saved_image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image = cv2.imread(saved_image)
            predicted_emotion = DeepFace.analyze(image)
            emotion = predicted_emotion['dominant_emotion']
            race = predicted_emotion['dominant_race']
            age = predicted_emotion['age']
            gender = predicted_emotion['gender']
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = haar.detectMultiScale(gray, 1.3, 5)
            print(face)
            for (x,y,w,h) in face:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), image)
            return render_template('result.html',
                                    filename=filename,
                                    image=saved_image,
                                    predicted_emotion=emotion,
                                    predicted_race=race,
                                    predicted_age=age,
                                    predicted_gender=gender)
    else:
        return "USE POST!"

if __name__ == '__main__':
    app.run(debug=True)