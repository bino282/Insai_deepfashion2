import json
from PIL import Image
import numpy as np
import os

base_dir="../local/street2shop/"
dataset = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}

dataset['categories'].append({
    'id': 1,
    'name': "bags",
    'supercategory': ""
})
dataset['categories'].append({
    'id': 2,
    'name': "belts",
    'supercategory': ""
})
dataset['categories'].append({
    'id': 3,
    'name': "dresses",
    'supercategory': ""
})
dataset['categories'].append({
    'id': 4,
    'name': "eyewear",
    'supercategory': ""
})
dataset['categories'].append({
    'id': 5,
    'name': "footwear",
    'supercategory': ""
})
dataset['categories'].append({
    'id': 6,
    'name': "hats",
    'supercategory': ""
})
dataset['categories'].append({
    'id': 7,
    'name': "leggings",
    'supercategory': ""
})
dataset['categories'].append({
    'id': 8,
    'name': "outerwear",
    'supercategory': ""
})
dataset['categories'].append({
    'id': 9,
    'name': "pants",
    'supercategory': ""
})
dataset['categories'].append({
    'id': 10,
    'name': "skirts",
    'supercategory': ""
})
dataset['categories'].append({
    'id': 11,
    'name': "tops",
    'supercategory': ""
})
cat2id = {"bags":1,"belts":2,"dresses":3,"eyewear":4,"footwear":5,"hats":6,"leggings":7,"outerwear":8,"pants":9,"skirts":10,"tops":11}
sub_index = 0 # the index of ground truth instance
list_file = os.listdir(os.path.join(base_dir,"meta/json"))
for file_name in list_file:
    if "train" not in file_name:
        continue
    with open(os.path.join(base_dir,"meta/json",file_name),'r') as f:
        temp = json.loads(f.read())
    for p in temp:
        image_name = os.path.join(base_dir,'images/')+str(p["photo"]).zfill(9)+".jpeg"
        try:
            imag = Image.open(image_name)
        except:
            continue
        width, height = imag.size
        dataset['images'].append({
                'coco_url': '',
                'date_captured': '',
                'file_name': str(p["photo"]).zfill(9)+".jpeg",
                'flickr_url': '',
                'id': p["photo"],
                'license': 0,
                'width': width,
                'height': height
            })
        sub_index = sub_index + 1
        box = p['bbox']
        w = box["width"]
        h = box["height"]
        x_1 = box["top"]
        y_1 = box["left"]
        bbox=[x_1,y_1,w,h]
        cat = cat2id[file_name.split('_')[-1].split('.')[0]]
        dataset['annotations'].append({
                        'area': w*h,
                        'bbox': bbox,
                        'category_id': cat,
                        'id': sub_index,
                        'image_id': p["photo"],
                        'iscrowd': 0
                    })


json_name = os.path.join(base_dir,'street2shop.json')
with open(json_name, 'w') as f:
  json.dump(dataset, f,indent=4)
