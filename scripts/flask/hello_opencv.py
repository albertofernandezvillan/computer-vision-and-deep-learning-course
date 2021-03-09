"""
Minimal example to introduce the use of OpenCV capabilities in a Flask application in Colab
"""

# Import required packages:
import cv2
from flask import Flask, request, make_response
from flask_ngrok import run_with_ngrok 
import numpy as np
import urllib.request

app = Flask(__name__)
run_with_ngrok(app)

@app.route("/") 
def home(): 
    return "Flask app server is running"

@app.route('/canny', methods=['GET'])
def canny_processing():
    # Get the image:
    with urllib.request.urlopen(request.args.get('url')) as url:
        image_array = np.asarray(bytearray(url.read()), dtype=np.uint8)

    # Convert the image to OpenCV format:
    img_opencv = cv2.imdecode(image_array, -1)

    # Convert image to grayscale:
    gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)

    # Perform canny edge detection:
    edges = cv2.Canny(gray, 100, 200)

    # Compress the image and store it in the memory buffer:
    retval, buffer = cv2.imencode('.jpg', edges)

    # Build the response:
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'

    # Return the response:
    return response


if __name__ == "__main__":
    app.run()
