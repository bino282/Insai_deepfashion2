import cv2
import requests
img = cv2.imread('test.jpeg')
files = {'image': open('test.jpeg', 'rb')}
result = requests.post("http://ec2-18-138-214-93.ap-southeast-1.compute.amazonaws.com:8081/detect_obj", files=files).json()
print(result)
for p in result["rois"]:
    cv2.rectangle(img, (int(p[0]),int(p[1])), (int(p[2]),int(p[3])),(255,0,0),2)
cv2.imshow("image",img)
cv2.waitKey(0)