import os
import cv2
import time
import json
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response, jsonify, abort

import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()

def platePattern(string):
    '''Returns true if passed string follows
    the pattern of indian license plates,
    returns false otherwise.
    '''
    if len(string) < 9 or len(string) > 10:
        return False
    elif string[:2].isalpha() == False:
        return False
    elif string[2].isnumeric() == False:
        return False
    elif string[-4:].isnumeric() == False:
        return False
    elif string[-6:-4].isalpha() == False:
        return False
    else:
        return True

app = Flask(__name__)

@app.get("/")
def read_root():
    return {"message": "Welcome to Licence plate recognition"}

@app.route('/detections', methods=['GET', 'POST'])
def get_detections():
    # raw_image = []
    # images = request.files.getlist("images")
    
    # print(images)
    # for image in images:
    #     image_name = image.filename
    #     image.save(os.path.join(os.getcwd(), image_name))
    #     img_raw = tf.image.decode_image(
    #         open(image_name, 'rb').read(), channels=3)
    #     raw_image.append(img_raw)
    # licence_plate_image = raw_image[0]

    json_data = request.get_json() #Get the POSTed json
    # dict_data = json.loads(json_data) #Convert json to dictionary
    img = json_data["img"] #Take out base64 str
    img = base64.b64decode(img) #Convert image data converted to base64 to original binary data# bytes
    jpg_as_np = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    licence_plate_image = img
    
    response = []
    t1 = time.time()
    licence_plate_image = cv2.cvtColor(np.array(licence_plate_image), cv2.COLOR_BGR2RGB)
    prediction_groups = pipeline.recognize([licence_plate_image]) # Text detection and recognition on license plate
    Licence_Number = ''
    for j in range(len(prediction_groups[0])):
        Licence_Number = Licence_Number+ prediction_groups[0][j][0].upper()
    t2 = time.time()
    print('time: {}'.format(t2 - t1))
    # print(Licence_Number)

    response = {"Licence plate": Licence_Number}
    try:
        return jsonify(response), 200
    except FileNotFoundError:
        abort(404)


app.run()


# "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"
# "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"