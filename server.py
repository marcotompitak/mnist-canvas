# Runs on 0.0.0.0, port specified by Heroku
# Allows CORS

from flask import Flask, url_for, request
from flask_cors import CORS, cross_origin
import io
import numpy as np
import cv2
import keras
import base64
from keras.models import load_model

# Setting up the Flask app
app = Flask(__name__)

# Allow Cross-Origin Resource Sharing
cors = CORS(app)

# Serve a canvas interface on /
@app.route('/')
def api_root():
    return app.send_static_file('interface.html')

@app.route('/predict', methods = ['POST'])
def api_predict():
    # First, getting the image into a usable format. Credit:
    # https://gist.github.com/mjul/32d697b734e7e9171cdb
    
    # Get the uploaded image
    img = request.files['file']
    
    # Store in memory
    in_memory_file = io.BytesIO()
    img.save(in_memory_file)
    
    # Convert to array representing black-and-white image
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 0)/255
    
    # Reshape for the model
    data = img.reshape(-1, 28*28)
    
    # The model is trained on white-on-black images. If a white-on-black image
    # is used, it will likely give incorrect results. Here is a basic fix to
    # try to detect black-on-white images and convert them. Probably not fool-proof.
    if ( data.mean() > 0.5 ):
        data = ValueInvert(data)
        
    # Generate prediction
    pred = model.predict_classes(data)[0]
    
    # Provide response
    return "The image you uploaded shows a " + str(pred) + ".\n"

@app.route('/post-data-url', methods = ['POST'])
@cross_origin()
def api_predict_from_dataurl():
    # In this case, we read the image data from a base64 data URL
    imgstring = request.form.get('data')
    
    # Stripping the metadata
    imgstring = imgstring.replace("data:image/png;base64,", "")
    
    # Decoding
    imgdata = base64.b64decode(imgstring)
    
    # Unfortunately I have not yet figured out how to get this data into
    # a usable format directly, without first saving to file
    filename = 'canvas_image.png'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    
    # Reading the image into OpenCV
    img = cv2.imread('canvas_image.png', 0)
    
    # Resize to fit the neural network
    imgsmall = cv2.resize(img, (28,28))
    
    # Save the resized image for manual checking
    cv2.imwrite("resized.png", imgsmall)
    
    # Reshape for the model
    data = imgsmall.reshape(-1, 28*28)/255
    
    # Convert to white-on-black
    data = ValueInvert(data)
    
    # Generate prediction
    pred = model.predict_classes(data)[0]
    
    print("Prediction requested! Returned " + str(pred))
    
    # Return the prediction
    return str(pred)


# Takes an array, assumed to contain values between 0 and 1, and inverts
# those values with the transformation x -> 1 - x.
def ValueInvert(array):
    # Flatten the array for looping
    flatarray = array.flatten()
    
    # Apply transformation to flattened array
    for i in range(flatarray.size):
        flatarray[i] = 1 - flatarray[i]
        
    # Return the transformed array, with the original shape
    return flatarray.reshape(array.shape)




# Load in the saved neural network
model = load_model('MNIST_NN.h5')

# Start flask app
if __name__ == '__main__':
    from os import environ
    app.run(debug=False, port=environ.get("PORT", 5000), host='0.0.0.0')
