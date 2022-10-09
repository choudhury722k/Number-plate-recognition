import cv2
import time
import json
import base64
import random
import requests
import colorsys
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from tflite_runtime.interpreter import Interpreter

# import keras_ocr
# pipeline = keras_ocr.pipeline.Pipeline()

iou = 0.45
threshold = 0.40
input_size = 416
output = './results'
images = './inputs/car.jpg'
weights = './data/yolov4-lpd.tflite'
NUM_CLASS = 1
video_path = '/home/soumya/Projects/Number Plate Recognition/inputs/demo1.mp4'

# video_path = 0

show_label=False
dont_show_video = False
dont_show_image = False
dont_show_licence_plate = False

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

def drawText(img, plates):
    string  = 'plates detected :- ' + plates[0]
    for i in range(1, len(plates)):
        string = string + ', ' + plates[i]
    
    font_scale = 2
    font = cv2.FONT_HERSHEY_PLAIN

    (text_width, text_height) = cv2.getTextSize(string, font, fontScale=font_scale, thickness=1)[0]
    box_coords = ((1, 30), (10 + text_width, 20 - text_height))
    
    cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(img, string, (5, 25), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)

def image_input():
    interpreter = Interpreter(model_path=weights)
    
    image_path = images
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    boxes = pred[0]
    pred_conf = pred[1]

    scores_max = pred_conf.reshape(pred_conf.shape[0], pred_conf.shape[1])
    mask = scores_max >= threshold
    class_boxes = boxes[mask]
    conf = pred_conf[mask]

    class_boxes = np.reshape(class_boxes, [1, class_boxes.shape[0], class_boxes.shape[1]])
    pred_conf = np.reshape(conf, [1, conf.shape[0], conf.shape[1]])

    box_xy, box_wh = np.split(class_boxes, 2, 2)
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = np.array([input_size, input_size])

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape

    boxes = np.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    
    max_th = 0
    bbox = []
    for i in range(0, len(pred_conf[0])):
        if pred_conf[0][i] > max_th and pred_conf[0][i] < 1.0:
            max_th = pred_conf[0][i]
            bbox = boxes[0][i]

    class_names = "Licence_plate"
    num_classes = len(class_names)
    image_h, image_w, _ = original_image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    coor = bbox
    coor[0] = int(coor[0] * image_h)
    coor[2] = int(coor[2] * image_h)
    coor[1] = int(coor[1] * image_w)
    coor[3] = int(coor[3] * image_w)
    crop = coor

    fontScale = 0.5
    score = max_th[0]
    
    bbox_color = colors[0]
    bbox_thick = int(0.6 * (image_h + image_w) / 600)
    c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
    cv2.rectangle(original_image, c1, c2, bbox_color, bbox_thick)

    img = Image.fromarray(original_image.astype(np.uint8))
    if not dont_show_image:
        img.show()
    
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    box = crop
    licence_plate_image = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
    
    # prediction_groups = pipeline.recognize([licence_plate_image]) # Text detection and recognition on license plate
    # string = ''
    # for j in range(len(prediction_groups[0])):
    #    string = string+ prediction_groups[0][j][0].upper()
    # print(string)    
    
    licence_plate_image = Image.fromarray(licence_plate_image.astype(np.uint8))
    if not dont_show_licence_plate:
        licence_plate_image.show()

    buffered = BytesIO()
    licence_plate_image.save(buffered, format="JPEG")
    img_byte = buffered.getvalue() 
    img_base64 = base64.b64encode(img_byte)

    img_str = img_base64.decode('utf-8') 
    # print(img_str)

    files = {"text":"Licence plate",
            "img":img_str}

    # r = requests.post("http://127.0.0.1:5000", json=json.dumps(files)) #POST to server as json
    # print(r.json())

    date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
    print(date)

    data = {"Licence plate":"",
            "time":date}

def video_input():
    interpreter = Interpreter(model_path=weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)

    plates = []

    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    while True:

        return_value, frame = vid.read()

        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        # cv2.imshow('Frame',frame)

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        boxes = pred[0]
        pred_conf = pred[1]

        scores_max = pred_conf.reshape(pred_conf.shape[0], pred_conf.shape[1])
        mask = scores_max >= threshold
        class_boxes = boxes[mask]
        conf = pred_conf[mask]

        class_boxes = np.reshape(class_boxes, [1, class_boxes.shape[0], class_boxes.shape[1]])
        pred_conf = np.reshape(conf, [1, conf.shape[0], conf.shape[1]])

        box_xy, box_wh = np.split(class_boxes, 2, 2)
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        input_shape = np.array([input_size, input_size])

        box_mins = (box_yx - (box_hw / 2.)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.)) / input_shape

        boxes = np.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)

        max_th = 0
        bbox = []
        for i in range(0, len(pred_conf[0])):
            if pred_conf[0][i] > max_th and pred_conf[0][i] < 1.0:
                max_th = pred_conf[0][i]
                bbox = boxes[0][i]

        class_names = "Licence_plate"
        num_classes = len(class_names)
        image_h, image_w, _ = frame.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        if max_th > 0.9:

            coor = bbox
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)
            crop = coor

            fontScale = 0.5
            score = max_th[0]
        
            bbox_color = colors[0]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
            cv2.rectangle(frame, c1, c2, bbox_color, bbox_thick)

            img = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            box = crop
            licence_plate_image = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]

            # prediction_groups = pipeline.recognize([licence_plate_image]) # Text detection and recognition on license plate
            # Licence_Number = ''
            # for j in range(len(prediction_groups[0])):
            #     Licence_Number = Licence_Number+ prediction_groups[0][j][0].upper()
            # print(Licence_Number)

            # if platePattern(Licence_Number) == True and Licence_Number not in plates:
            #    plates.append(Licence_Number)
        
        # if len(plates) > 0:
        #     drawText(result, plates)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not dont_show_video:
            cv2.imshow("result", result)

        if cv2.waitKey(25) & 0xFF == ord('q'): break
    
    vid.release()
    cv2.destroyAllWindows()


image_input()