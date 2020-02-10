# import keras
import keras
import os,sys
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"
# import keras_retinanet
from .. import models
from ..utils.image import read_image_bgr, preprocess_image, resize_image
from ..utils.visualization import draw_box, draw_caption
from ..utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import argparse

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

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
        if (labels[i] not in show_lb) and scores[i]>score_thresh:
            show_lb.append(labels[i])
            show_box.append(boxes[i])
            show_scores.append(scores[i])
        else:
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

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('../../local/resnet50_csv_15.h5')

parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
parser.add_argument('--model_path',default='../../local/resnet50_csv_15.h5')
parser.add_argument('--image_path',default='ezgif.jpg')
parser.add_argument('--thresh',default=0.1)
args = parser.parse_args()
def main():

    id2name= {}
    with open("name2id.txt",'r',encoding="utf-8") as lines:
        for line in lines:
            tmp = line.strip().split(',')
            id2name[tmp[1]]=tmp[0]
    print(id2name)
    # load retinanet model
    model_path = args.model_path
    model = models.load_model(model_path, backbone_name='resnet50')

    # if the model is not converted to an inference model, use the line below
    # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
    model = models.convert_model(model)

    # print(model.summary())
    # load image
    ori_image = cv2.imread(args.image_path)
    image = read_image_bgr(args.image_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    # visualize detections
    show_box,show_scores,show_lb = select_box(boxes[0], scores[0], labels[0],score_thresh=args.thresh)
    for box, score, label in zip(show_box,show_scores,show_lb):
        print(box)
        print(str(label)+" : "+str(score))
        # scores are sorted so we can break
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(ori_image, b, color=color)
        caption = "{} {:.3f}".format(id2name[str(label)], score)
        draw_caption(ori_image, b, caption)
    cv2.imwrite("result.jpeg",ori_image)
if __name__ == '__main__':
    main()
