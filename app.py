# import keras
import keras
import os,sys
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from tensorflow.python.keras.backend import set_session

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import argparse
import tensorflow as tf
import json

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """

    # determine the coordinates of the intersection rectangle
    bb1 = bb1.flatten().tolist()
    bb2 = bb2.flatten().tolist()
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def select_box(boxes,scores,labels,score_thresh=0.5):
    show_lb = []
    show_box = []
    show_scores = []
    for i in range(len(labels)):
        # if (labels[i] not in show_lb) and scores[i]>score_thresh:
        if scores[i]>score_thresh:
            show_lb.append(labels[i])
            show_box.append(boxes[i])
            show_scores.append(scores[i])
        else:
            continue
            if(scores[i]>score_thresh):
                index = show_lb.index(labels[i])
                if(scores[i]>=show_scores[index]):
                    show_box[index] = boxes[i]
                    show_scores[index] = scores[i]

    show_lb_1 = [show_lb[0]]
    show_box_1 = [show_box[0]]
    show_scores_1 = [show_scores[0]]
    for i in range(1,len(show_box)):
        check = 0
        for j in range(len(show_box_1)):
            if get_iou(show_box[i],show_box_1[j]) > 0.6:
                if(show_scores[i] >= show_scores_1[j]):
                    show_box_1[j] = show_box[i]
                    show_scores_1[j] = show_scores[i]
                    show_lb_1[j] = show_lb[i]
                check = 1
        if(check==0):
            show_box_1.append(show_box[i])
            show_lb_1.append(show_lb[i])
            show_scores_1.append(show_scores[i])
    return show_box_1,show_scores_1,show_lb_1
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

from flask import Flask, url_for, send_from_directory, request,render_template
import logging, os
from werkzeug.utils import secure_filename

app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/static/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath
@app.route("/get_detection/<float:thresh>", methods=["POST"])
def get_box(thresh):
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)

        ori_image = cv2.imread(saved_path)
        image = read_image_bgr(saved_path)
        # image,scale = resize_image(image,400,600)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        with graph.as_default():
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale
        # visualize detections
        try:
            show_box,show_scores,show_lb = select_box(boxes[0], scores[0], labels[0],score_thresh=thresh)
        except:
            return  {"boxes":[],"scores":[],"labels":[]}
        result = {"boxes":[],"scores":[],"labels":[]}
        for box, score, label in zip(show_box,show_scores,show_lb):
            tmp = {}
            Rx = 400/ori_image.shape[1]
            Ry = 600/ori_image.shape[0]
            box_tmp = box.flatten().tolist()
            tmp["width"] = Rx*(box_tmp[2]-box_tmp[0])
            tmp["height"] = Ry*(box_tmp[3] - box_tmp[1])
            tmp["x"] = Rx*box_tmp[0]
            tmp["y"] = Ry*box_tmp[1]
            result["boxes"].append(tmp)
            result["scores"].append(str(score))
            result["labels"].append(id2name[str(label)])
    return json.dumps(result)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--model_path',default='../../local/resnet50_csv_15.h5')
    parser.add_argument('--image_path',default='ezgif.jpg')
    parser.add_argument('--thresh',default=0.1,type=float)
    args = parser.parse_args()
    id2name= {}
    with open("name2id.txt",'r',encoding="utf-8") as lines:
        for line in lines:
            tmp = line.strip().split(',')
            id2name[tmp[1]]=tmp[0]
    print(id2name)
    sess = get_session()
    graph = tf.get_default_graph()
    keras.backend.tensorflow_backend.set_session(sess)
    model_path = args.model_path
    model = models.load_model(model_path, backbone_name='resnet50')
    model = models.convert_model(model)
    app.run(host='0.0.0.0',port=8081,threaded=True)

