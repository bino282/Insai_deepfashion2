import numpy, redis, json
from flask import Flask, render_template
from flask import Flask, url_for, send_from_directory, request,render_template
import requests
import logging
from visualize import *
from werkzeug import secure_filename
import os
import time
app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/static/uploads/'.format(PROJECT_HOME)
IMAGE_FOLDER = '{}/static/images/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
def create_new_folder(local_dir):
	newpath = local_dir
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	return newpath

@app.route('/upload/<float:thresh>', methods = ['POST'])
def upload_image(thresh):
	if request.method == 'POST' and request.files['image-file']:
		app.logger.info(app.config['UPLOAD_FOLDER'])
		img = request.files['image-file']
		img_name = secure_filename(img.filename)
		create_new_folder(app.config['UPLOAD_FOLDER'])
		saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
		app.logger.info("saving {}".format(saved_path))
		img.save(saved_path)
		ori_image = cv2.imread(saved_path)
		url = 'http://ec2-18-138-214-93.ap-southeast-1.compute.amazonaws.com:8081/get_detection/{}'.format(thresh)
		files = {'image': open(saved_path, 'rb')}
		r = requests.post(url, files=files)
		show_box = r.json()["boxes"]
		show_scores = r.json()["scores"]
		show_lb = r.json()["labels"]
		lb2id = {}
		for lb in show_lb:
			if lb not in lb2id:
				lb2id[lb]=len(lb2id)
		for box, score, label in zip(show_box,show_scores,show_lb):
			box = [float(b) for b in box]		   
			color = label_color(lb2id[label])
			box= np.array(box)		
			b = box.astype(int)
			draw_box(ori_image, b, color=color)
			caption = "{} {:.3f}".format(label, float(score))
			draw_caption(ori_image, b, caption)
		img_name = "result-{}.jpeg".format(time.time())
		save_path = os.path.join(app.config['IMAGE_FOLDER'],img_name)
		cv2.imwrite(save_path,ori_image)
		return json.dumps({"status":1,"name":img_name})
@app.route('/home',methods =['GET'])
def render_analytics():
	return render_template('analytics.html', title = 'Recommendation Testing')
if __name__ == "__main__":
	app.run(host ='0.0.0.0', port = 8801)