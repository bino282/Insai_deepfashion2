import json
from PIL import Image
import numpy as np
import os
import traceback
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
with open("cat2id.txt","w") as fr:
    for e in cat2id:
        fr.write(e+','+str(cat2id[e]-1))
        fr.write("\n")
sub_index = 0 # the index of ground truth instance
list_file = os.listdir(os.path.join(base_dir,"meta/json"))
# fw = open("error.txt",'w')
f_csv = open("test.csv","w")
for file_name in list_file:
    if "test" not in file_name:
        continue
    print(file_name)
    with open(os.path.join(base_dir,"meta/json",file_name),'r') as f:
        temp = json.loads(f.read())
    print(len(temp))
    for p in temp:
        row = []
        try:
            image_name = os.path.join(base_dir,'images/')+str(p["photo"]).zfill(9)+".jpeg"
            imag = Image.open(image_name)
            row.append(image_name)
        except:
            # traceback.print_exc()
            try:
                image_name = os.path.join(base_dir,'images/')+str(p["photo"]).zfill(9)+".png"
                imag = Image.open(image_name)
                row.append(image_name)
            except:
                # traceback.print_exc()
                # fw.write(str(p["photo"]).zfill(9))
                # fw.write("\n")
                continue
        width, height = imag.size
        dataset['images'].append({
                'coco_url': '',
                'date_captured': '',
                'file_name': image_name.split('/')[-1],
                'flickr_url': '',
                'id': p["photo"],
                'license': 0,
                'width': width,
                'height': height
            })
        sub_index = sub_index + 1
        box = p['bbox']
        w = int(box["width"])
        h = int(box["height"])
        x_1 = int(box["left"])
        y_1 = int(box["top"])
        if(w==0 or h==0):
            row = row+["","","","",""]
        else:
            bbox=[x_1,y_1,w,h]
            row = row+[str(x_1),str(y_1),str(x_1+w),str(y_1+h)]
            row.append(file_name.split('_')[-1].split('.')[0])
        cat = cat2id[file_name.split('_')[-1].split('.')[0]]
        dataset['annotations'].append({
                        'area': w*h,
                        'bbox': bbox,
                        'category_id': cat,
                        'id': sub_index,
                        'image_id': p["photo"],
                        'iscrowd': 0
                    })
        f_csv.write(",".join(row))
        f_csv.write("\n")
f_csv.close()
#fw.close()
print(len(dataset["images"]))
json_name = os.path.join(base_dir,'street2shop_train.json')
with open(json_name, 'w') as f:
  json.dump(dataset, f,indent=4)
