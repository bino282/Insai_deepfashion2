import requests
import cv2
url = 'http://localhost:5432/get_detection/0.2'
files = {'image': open('test.jpg', 'rb')}
r = requests.post(url, files=files)
print(r.json())
exit()
img = cv2.imread("test.jpg")
for p in r.json()['rois']:
    print(p)
    if(p[4]> 0.6):
        cv2.rectangle(img, (int(p[0]),int(p[1])), (int(p[2]), int(p[3])),(255,0,0),2)
cv2.imshow('test',img)
cv2.waitKey(0)