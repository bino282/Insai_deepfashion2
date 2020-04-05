
import json
import os
import cv2
import numpy as np
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color
from keras_maskrcnn.utils.visualization import draw_mask
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
import time
import keras
import matplotlib.pyplot as plt
from keras.utils import plot_model

from flask import Flask, url_for, send_from_directory, request,render_template
import logging, os
from werkzeug.utils import secure_filename
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/static/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from keras_maskrcnn import models
def load_image(path):
    return read_image_bgr(path)


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

        image = load_image(saved_path)
        image_shape = image.shape
        image = preprocess_image(image)
        image, scale = resize_image(image)
        with graph.as_default():
            outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes  = outputs[-4][0]
        scores = outputs[-3][0]
        labels = outputs[-2][0]
        boxes_final = []
        score_final = []
        label_final = []
        # correct for image scale
        boxes /= scale
        for box, score, label in zip(boxes, scores, labels):
            if score < thresh:
                continue
            else:
                boxes_final.append(box)
                score_final.append(score)
                label_final.append(label)
        result = {"boxes":[],"scores":[],"labels":[]}
        for box, score, label in zip(boxes_final,score_final,label_final):
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
            result["labels"].append(labels_to_names[label])
    return json.dumps(result)
if __name__ == "__main__":
    model_path = "../../local/resnet50_modanet.h5"
    labels_to_names = {0: 'bag', 1: 'belt', 2: 'boots', 3: 'footwear', 4: 'outer', 5: 'dress', 6: 'sunglasses', 7: 'pants', 8: 'top', 9: 'shorts', 10: 'skirt', 11: 'headwear', 12: 'scarf/tie'}
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()
    keras.backend.tensorflow_backend.set_session(sess)
    model = models.load_model(model_path, backbone_name='resnet50')
    app.run(host='0.0.0.0',port=8081,threaded=True)