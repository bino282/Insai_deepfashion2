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
# with open(os.path.expanduser('~')+ '/.maskrcnn-modanet/' + 'savedvars.json') as f:
# 	savedvars = json.load(f)
# path = savedvars['datapath']

# ann_path = path + "datasets/coco/annotations/"
# ann_orig_path = path + 'datasets/modanet/annotations/'

coco_path = "datasets/coco/"

labels_to_names = {0: 'bag', 1: 'belt', 2: 'boots', 3: 'footwear', 4: 'outer', 5: 'dress', 6: 'sunglasses', 7: 'pants', 8: 'top', 9: 'shorts', 10: 'skirt', 11: 'headwear', 12: 'scarf/tie'}
from keras_maskrcnn import models

model_path = "../../local/resnet50_modanet.h5"
model = models.load_model(model_path, backbone_name='resnet50')
print(model.summary())
#plot_model(model, to_file='./model.png')

# classification_feature_extractor = keras.Model(inputs = model.inputs, outputs= model.get_layer(name="classification_submodel").get_layer("pyramid_classification_3").output)

# print(classification_feature_extractor.summary())
# exit()
def load_image(path):
    return read_image_bgr(path)


image = load_image("test2.jpg")
draw = image.copy()
#draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
image_shape = image.shape
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

boxes  = outputs[-4][0]
scores = outputs[-3][0]
labels = outputs[-2][0]
masks  = outputs[-1][0]

# correct for image scale
boxes /= scale

# visualize detections
for box, score, label, mask in zip(boxes, scores, labels, masks):
    if score < 0.5:
        break

    color = label_color(label)  
    b = box.astype(int)
    draw_box(draw, b, color=color)
    
    mask = mask[:, :, label]
    #draw_mask(draw, b, mask, color=label_color(label))
    
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
    
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()