import numpy as np
from flask import Flask, request, jsonify, render_template, flash
import werkzeug     #file handling 
from tensorflow import keras
from keras.utils import load_img
from keras.applications.vgg19 import preprocess_input
import parselmouth
from convert import measurePitch
import joblib

# Loads the random forest model
loaded_rf = joblib.load("./rf.joblib")

# Reads the contents of the JSON file and stores it.
with open('m.json', 'r') as json_file:
    m1 = json_file.read()
m = keras.models.model_from_json(m1)
model = keras.models.model_from_json(m1)  # Load the saved model's m.json
m.load_weights('m.h5')
model.load_weights('m.h5')  # Load weight
inputShape = (224, 224)
preprocess = preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_SOUND = {'wav', 'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/')  # Home page route
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'sound' not in request.files:
        flash('No file found')
        return "Error"

    imagefile = request.files['image']  #making sure file formats are correct
    soundfile = request.files['sound']

    if imagefile and allowed_file(imagefile.filename, ALLOWED_EXTENSIONS) and soundfile and allowed_file(soundfile.filename, ALLOWED_EXTENSIONS_SOUND):
        image_filename = werkzeug.utils.secure_filename(imagefile.filename)     #sanitizing file names
        sound_filename = werkzeug.utils.secure_filename(soundfile.filename)

        imagefile.save(f'./{UPLOAD_FOLDER}/{image_filename}')
        soundfile.save(f'./{UPLOAD_FOLDER}/{sound_filename}')

        # Process the image
        img = load_img(f'./{UPLOAD_FOLDER}/{image_filename}',
                       target_size=(224, 224))
        img = np.array(img)
        img = preprocess(img)
        p1 = m.predict(np.array([img]))
        p1_label = 'Patient' if p1 > 0.5 else 'Control'

        # Process the sound
        sound = parselmouth.Sound(f'./{UPLOAD_FOLDER}/{sound_filename}')
        # Measure pitch
        (localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer,
         apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, ac, nth, htn, median_pitch, meanF0, stdevF0,
         min_pitch, max_pitch, n_pulses, n_periods, mean_period, standard_deviation_period,
         fraction_unvoiced_frames, num_voice_breaks, degree_voice_breaks) = measurePitch(sound, 75, 500, "Hertz")
        X = [localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer,
             apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, ac, nth, htn, median_pitch, meanF0, stdevF0,
             min_pitch, max_pitch, n_pulses, n_periods, mean_period, standard_deviation_period,
             fraction_unvoiced_frames, num_voice_breaks, degree_voice_breaks]
        p2 = loaded_rf.predict(np.array([X]))[0]
        p2_label = 'Control' if p2 == 0 else "Parkinson's"

        # Calculate the final diagnosis
        if p1_label == 'Patient' and p2_label == 'Parkinson\'s':
            diagnosis = 'Parkinson\'s'
        else:
            diagnosis = 'Healthy'

        return render_template('diagnosis.html', diagnosis=diagnosis)
    else:
        return "Error"


if __name__ == "__main__":
    app.run(debug=True)
