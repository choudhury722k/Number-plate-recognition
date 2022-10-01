# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

import cv2
import time
import numpy as np
from PIL import Image

import Core.utils as utils
from Core.config import cfg
from Core.yolov4 import YOLOv4, decode, filter_boxes

import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants

iou = 0.45
score = 0.25
input_size = 416
output = './results'
images = './inputs/car.jpg'
weights = './data/yolov4-lpd.tflite'
XYSCALE = cfg.YOLO.XYSCALE
STRIDES = np.array(cfg.YOLO.STRIDES)
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
# video_path = '/home/soumya/Number Plate Recognition/inputs/demo1.mp4'

video_path = 0

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
    '''Draws recognized plate numbers on the
    top-left side of frame
    '''
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
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    interpreter = tf.lite.Interpreter(model_path=weights)
    
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
    print(input_details)
    print(output_details)
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score)

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())

    # custom allowed classes (uncomment line below to allow detections for only people)
    #allowed_classes = ['person']

    image = utils.draw_bbox(original_image, pred_bbox, allowed_classes = allowed_classes)
    image = Image.fromarray(image.astype(np.uint8))
    if not dont_show_image:
        image.show()

    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    box = pred_bbox[0][0][0]
    licence_plate_image = image[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
    print(type(licence_plate_image))

    prediction_groups = pipeline.recognize([licence_plate_image]) # Text detection and recognition on license plate
    string = ''
    for j in range(len(prediction_groups[0])):
        string = string+ prediction_groups[0][j][0].upper()
    print(string)

    licence_plate_image = Image.fromarray(licence_plate_image.astype(np.uint8))
    if not dont_show_licence_plate:
        licence_plate_image.show()

def video_input():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    interpreter = tf.lite.Interpreter(model_path=weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

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
        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                        input_shape=tf.constant([input_size, input_size]))

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        
        image = utils.draw_bbox(frame, pred_bbox)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        img = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        box = pred_bbox[0][0][0]
        licence_plate_image = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]

        # if(box[0] > 0 and box[1] > 0):
        #     prediction_groups = pipeline.recognize([licence_plate_image]) # Text detection and recognition on license plate
        #     Licence_Number = ''
        #     for j in range(len(prediction_groups[0])):
        #         Licence_Number = Licence_Number+ prediction_groups[0][j][0].upper()
        #     print(Licence_Number)

        #     if platePattern(Licence_Number) == True and Licence_Number not in plates:
        #         plates.append(Licence_Number)
        
        # if len(plates) > 0:
        #     drawText(frame, plates)

        if not dont_show_video:
            cv2.imshow("result", result)
        
        if cv2.waitKey(25) & 0xFF == ord('q'): break
    
    vid.release()
    cv2.destroyAllWindows()

# image_input()
video_input()
