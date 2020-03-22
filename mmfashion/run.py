from mmcv import Config
from mmcv.runner import load_checkpoint
import torch
from mmfashion.core import AttrPredictor
from mmfashion.models import build_retriever
from mmfashion.utils import get_img_tensor
import time
import datetime
import re
config = "configs/retriever_in_shop/global_retriever_vgg_loss_id.py"
checkpoint = "../../local/sota_imat_epoch_15.pth"

cfg = Config.fromfile(config)
cfg.model.pretrained = None
model = build_retriever(cfg.model)
load_checkpoint(model, checkpoint,map_location='cpu')
print('load checkpoint from {}'.format(checkpoint))
model.eval()
def get_vector(image_path):
    img_tensor = get_img_tensor(image_path, False)
    query_feat = model(img_tensor, landmark=None, return_loss=False)
    query_feat = query_feat.data.cpu().numpy()
    return query_feat


mongoUrl = "mongodb://ec2-18-138-214-93.ap-southeast-1.compute.amazonaws.com:27017"
from pymongo import MongoClient
client = MongoClient(mongoUrl)
db = client.IMGS_VECTOR.Product_Vector_v3
import boto3
import tempfile
s3 = boto3.resource('s3',region_name='ap-southeast-1')
bucket = s3.Bucket('vs-insai-storage')
tmp = tempfile.NamedTemporaryFile()
def writedb(filenames,batch_size):
    for i in range(len(filenames)//batch_size +1):
        batch_arr = []
        ids = []
        for filename in filenames[i*batch_size:(i+1)*batch_size]:
            print('FashionImage/{}'.format(filename))
            try:
                _object = bucket.Object('FashionImage/{}'.format(filename))
                with open(tmp.name,'wb') as f:
                    _object.download_fileobj(f)
                    img_vector = get_vector(tmp.name)
                    ids.append(filename.split('.')[0]+"-{}".format(i))
                    batch_arr.append(img_vector)
            except Exception as e:
                logger.info(e)
                continue
            del _object
        if(len(batch_arr)==0):
            continue
        dict_result = []
        for i in range(len(ids)):
            _tmp = {}
            _tmp['ID'] = ids[i]
            _tmp['features'] = batch_arr[i].tolist()
            _tmp['createdDate'] = str(datetime.datetime.now())
            dict_result.append(_tmp)
        for d in dict_result:
            db.replace_one({'ID':d['ID']},d,True)

def check_id():
    new_id = []
    list_key = db.find({},{"ID":1})
    ids = []
    ids_s3 = []
    for key in list_key:
        ids.append(key['ID'])
    for object_summary in bucket.objects.filter(Prefix="FashionImage"):
        names = object_summary.key.split('/')
        if(names[-1]!=""):
            ids_s3.append(names[-1])
    for e in ids_s3:
        if (e.split('.')[0] not in ids) and (e[-3:]=='jpg'):
            new_id.append(e)
    return new_id
    
import time
import logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
s_time = time.time()
new_id = check_id()
logging.info(str(len(new_id))+' new images...')
if(len(new_id)!=0):    
    writedb(new_id,batch_size=32)
    logger.info('Time : '+str(time.time() - s_time))
